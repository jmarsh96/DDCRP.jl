# ============================================================================
# NBMeanDispersionPopulation - Direct NegBin with population offsets, global r
# ============================================================================
#
# Model:
#   y_i | γ_k, r ~ NegBin(r, r/(r + P_i · γ_k))  for observation i in cluster k
#   γ_k ~ InverseGamma(γ_a, γ_b)                   (cluster rate, sampled via MH)
#   r   ~ Gamma(r_a, r_b)                           (global dispersion, sampled via MH)
#
# Mean for observation i in cluster k:  E[y_i] = P_i · γ_k
# Variance:                             Var[y_i] = P_i·γ_k + (P_i·γ_k)²/r
#
# NOTE: Analytical marginalisation of γ_k is not tractable in this direct NegBin
# formulation. For cluster S_k, the integral over γ_k with a Gamma/InverseGamma
# prior requires evaluating
#
#   ∫₀^∞ γ_k^{a-1} · e^{-b·γ_k} · ∏_{i∈S_k} (r + P_i·γ_k)^{-(r+y_i)} dγ_k
#
# which has no closed form when P_i differ across observations (even equal P_i
# leads to a Kummer U confluent hypergeometric, not a standard distribution).
# The only tractable marginalisation uses the Poisson–Gamma hierarchy, already
# implemented in NBPopulationRatesMarg.
#
# γ_k is therefore sampled via Metropolis-Hastings with a log-normal random walk.
#
# Parameters: c (assignments), γ_k (cluster rates), r (global dispersion)
# Requires: exposure/population data P_i via CountDataWithPopulation
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBMeanDispersionPopulation <: NegativeBinomialModel

Direct Negative Binomial model with population offsets.

The mean for observation i in cluster k is E[y_i] = P_i · γ_k, where γ_k is
a cluster-level rate and P_i is the population exposure. Dispersion r is shared
globally. Unlike NBPopulationRatesMarg, there are no latent Poisson rates; the
NegBin likelihood is used directly and γ_k is sampled via Metropolis-Hastings.

Requires CountDataWithPopulation (population/exposure vector P).

Parameters:
- γ_k: Cluster rates (InverseGamma prior)
- r:   Global dispersion parameter (Gamma prior)
- c:   Customer assignments (RJMCMC)
"""
struct NBMeanDispersionPopulation <: NegativeBinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    NBMeanDispersionPopulationState{T<:Real} <: AbstractMCMCState{T}

State for NBMeanDispersionPopulation model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `γ_dict::Dict{Vector{Int}, T}`: Table → cluster rate γ_k
- `r::T`: Global dispersion parameter
"""
mutable struct NBMeanDispersionPopulationState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    γ_dict::Dict{Vector{Int}, T}
    r::T
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBMeanDispersionPopulationPriors{T<:Real} <: AbstractPriors

Prior specification for NBMeanDispersionPopulation model.

# Fields
- `γ_a::T`: InverseGamma shape parameter for cluster rate γ_k
- `γ_b::T`: InverseGamma scale parameter for cluster rate γ_k
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NBMeanDispersionPopulationPriors{T<:Real} <: AbstractPriors
    γ_a::T
    γ_b::T
    r_a::T
    r_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    NBMeanDispersionPopulationSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for NBMeanDispersionPopulation model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples × n_obs)
- `γ::Matrix{T}`: Per-observation cluster rate (n_samples × n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
- `α_ddcrp::Vector{T}`: DDCRP concentration samples (n_samples)
- `s_ddcrp::Vector{T}`: DDCRP decay scale samples (n_samples)
"""
struct NBMeanDispersionPopulationSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    γ::Matrix{T}
    r::Vector{T}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
    y_imp::Matrix{Float64}
    missing_indices::Vector{Int}
end

# ============================================================================
# Trait Functions
# ============================================================================

is_marginalised(::NBMeanDispersionPopulation) = false
requires_population(::NBMeanDispersionPopulation) = true

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::NBMeanDispersionPopulationState) = (γ = state.γ_dict,)

"""
    fixed_dim_params(model::NBMeanDispersionPopulation, S_i, table_old, table_new, state, data, priors, opts)

Fixed-dimensional parameter update (NoUpdate strategy): keep existing cluster
rates unchanged after moving S_i between clusters. The Metropolis-Hastings
step for γ_k in update_params! will correct these values in subsequent iterations.
"""
function fixed_dim_params(
    ::NBMeanDispersionPopulation,
    S_i::Vector{Int},
    table_old::Vector{Int},
    table_new::Vector{Int},
    state::NBMeanDispersionPopulationState,
    data::CountDataWithPopulation,
    priors::NBMeanDispersionPopulationPriors,
    opts::MCMCOptions
)
    γ_depleted  = state.γ_dict[table_old]
    γ_augmented = state.γ_dict[table_new]
    return (γ = γ_depleted,), (γ = γ_augmented,), 0.0
end

"""
    sample_birth_params(model::NBMeanDispersionPopulation, ::PriorProposal, S_i, state, data, priors)

Sample a new cluster rate γ for a birth move from the prior InverseGamma(γ_a, γ_b).
"""
function sample_birth_params(
    ::NBMeanDispersionPopulation,
    ::PriorProposal,
    S_i::Vector{Int},
    state::NBMeanDispersionPopulationState,
    data::CountDataWithPopulation,
    priors::NBMeanDispersionPopulationPriors
)
    Q = InverseGamma(priors.γ_a, priors.γ_b)
    γ_new = rand(Q)
    return (γ = γ_new,), logpdf(Q, γ_new)
end

"""
    birth_params_logpdf(model::NBMeanDispersionPopulation, ::PriorProposal, params_old, S_i, state, data, priors)

Log-density of the birth proposal at params_old (used for the reverse death move).
"""
function birth_params_logpdf(
    ::NBMeanDispersionPopulation,
    ::PriorProposal,
    params_old::NamedTuple,
    S_i::Vector{Int},
    state::NBMeanDispersionPopulationState,
    data::CountDataWithPopulation,
    priors::NBMeanDispersionPopulationPriors
)
    return logpdf(InverseGamma(priors.γ_a, priors.γ_b), params_old.γ)
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBMeanDispersionPopulation, table, state, data, priors)

Compute log-contribution of a table using direct NegBin likelihood with
population offset: y_i ~ NegBin(r, r/(r + P_i · γ_k)).

Returns the NegBin log-likelihood summed over the table plus the
InverseGamma log-prior on γ_k.
"""
function table_contribution(
    ::NBMeanDispersionPopulation,
    table::AbstractVector{Int},
    state::NBMeanDispersionPopulationState,
    data::CountDataWithPopulation,
    priors::NBMeanDispersionPopulationPriors
)
    y = observations(data)
    P = population(data)
    γ = state.γ_dict[sort(table)]
    r = state.r

    obs_table = any(ismissing, y) ? [i for i in table if !ismissing(y[i])] : table

    # InverseGamma prior on γ_k
    log_prior_γ = logpdf(InverseGamma(priors.γ_a, priors.γ_b), γ)
    isempty(obs_table) && return log_prior_γ

    # NegBin log-likelihood: y_i ~ NegBin(r, r/(r + P_i * γ))
    log_lik = sum(negbin_logpdf(y[i], P[i] * γ, r) for i in obs_table)

    return log_lik + log_prior_γ
end

"""
    impute_y(::NBMeanDispersionPopulation, i, state, data, priors)

Draw a value for missing observation i from NegBin(r, r/(r+P_i*γ_k)) using the
current cluster rate and global dispersion.
"""
function impute_y(
    ::NBMeanDispersionPopulation,
    i::Int,
    state::NBMeanDispersionPopulationState,
    data::CountDataWithPopulation,
    priors::NBMeanDispersionPopulationPriors
)
    P = population(data)
    tables = table_vector(state.c)
    table = tables[findfirst(t -> i in t, tables)]
    γ = state.γ_dict[sort(table)]
    r = state.r
    P_i = P isa Int ? Float64(P) : Float64(P[i])
    m_i = P_i * γ
    p = r / (r + m_i)
    return rand(NegativeBinomial(r, p))
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBMeanDispersionPopulation, data, state, priors, log_DDCRP)

Compute full log-posterior for the direct NegBin population model.
"""
function posterior(
    model::NBMeanDispersionPopulation,
    data::CountDataWithPopulation,
    state::NBMeanDispersionPopulationState,
    priors::NBMeanDispersionPopulationPriors,
    log_DDCRP::AbstractMatrix
)
    log_prior_r = logpdf(Gamma(priors.r_a, 1/priors.r_b), state.r)

    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.γ_dict)) +
           ddcrp_contribution(state.c, log_DDCRP) +
           log_prior_r
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_γ!(model::NBMeanDispersionPopulation, state, data, priors; prop_sd=0.3)

Update all cluster rates via Metropolis-Hastings with log-normal random walk.
"""
function update_γ!(
    model::NBMeanDispersionPopulation,
    state::NBMeanDispersionPopulationState,
    data::CountDataWithPopulation,
    priors::NBMeanDispersionPopulationPriors;
    prop_sd::Float64 = 0.3
)
    for table in keys(state.γ_dict)
        update_γ_table!(model, table, state, data, priors; prop_sd=prop_sd)
    end
end

"""
    update_γ_table!(model::NBMeanDispersionPopulation, table, state, data, priors; prop_sd=0.3)

Update the cluster rate γ_k for a single table using a log-normal random walk.
"""
function update_γ_table!(
    model::NBMeanDispersionPopulation,
    table::Vector{Int},
    state::NBMeanDispersionPopulationState,
    data::CountDataWithPopulation,
    priors::NBMeanDispersionPopulationPriors;
    prop_sd::Float64 = 0.3
)
    γ_old = state.γ_dict[table]
    log_γ_can = log(γ_old) + randn() * prop_sd
    γ_can = exp(log_γ_can)

    γ_dict_can = copy(state.γ_dict)
    γ_dict_can[table] = γ_can

    state_can = NBMeanDispersionPopulationState(state.c, γ_dict_can, state.r)

    logpost_current   = table_contribution(model, table, state,     data, priors)
    logpost_candidate = table_contribution(model, table, state_can, data, priors)

    # Log-normal random walk: Jacobian = log(γ_can) - log(γ_old)
    log_accept_ratio = logpost_candidate - logpost_current + log_γ_can - log(γ_old)

    if log(rand()) < log_accept_ratio
        state.γ_dict[table] = γ_can
    end
end

"""
    update_r!(model::NBMeanDispersionPopulation, state, data, priors, tables; prop_sd=0.5)

Update global dispersion r via Metropolis-Hastings with Normal random walk.
"""
function update_r!(
    model::NBMeanDispersionPopulation,
    state::NBMeanDispersionPopulationState,
    data::CountDataWithPopulation,
    priors::NBMeanDispersionPopulationPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    r_can = rand(Normal(state.r, prop_sd))
    r_can <= 0 && return

    state_can = NBMeanDispersionPopulationState(state.c, state.γ_dict, r_can)

    logpost_current = sum(table_contribution(model, table, state,     data, priors) for table in tables) +
                      logpdf(Gamma(priors.r_a, 1/priors.r_b), state.r)
    logpost_candidate = sum(table_contribution(model, table, state_can, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1/priors.r_b), r_can)

    log_accept_ratio = logpost_candidate - logpost_current

    if log(rand()) < log_accept_ratio
        state.r = r_can
    end
end

"""
    update_params!(model::NBMeanDispersionPopulation, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters (γ_k and r). Assignment updates are handled
separately by `update_c!` in the main MCMC loop.
"""
function update_params!(
    model::NBMeanDispersionPopulation,
    state::NBMeanDispersionPopulationState,
    data::CountDataWithPopulation,
    priors::NBMeanDispersionPopulationPriors,
    tables::Vector{Vector{Int}},
    ::AbstractMatrix,
    opts::MCMCOptions
)
    if should_infer(opts, :γ)
        update_γ!(model, state, data, priors; prop_sd=get_prop_sd(opts, :γ))
    end

    if should_infer(opts, :r)
        update_r!(model, state, data, priors, tables; prop_sd=get_prop_sd(opts, :r))
    end
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::NBMeanDispersionPopulation, data, ddcrp_params, priors)

Create initial MCMC state. Cluster rates are initialised at the empirical
mean rate (y_i / P_i) within each initial cluster; r is set to 1.0.
"""
function initialise_state(
    ::NBMeanDispersionPopulation,
    data::CountDataWithPopulation,
    ddcrp_params::DDCRPParams,
    priors::NBMeanDispersionPopulationPriors
)
    y = observations(data)
    P = population(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)
    γ_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        rates = [y[i] / max(Float64(P[i]), 1.0) for i in table]
        γ_dict[sort(table)] = max(mean(rates), 1e-4)
    end
    return NBMeanDispersionPopulationState(c, γ_dict, 1.0)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBMeanDispersionPopulation, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBMeanDispersionPopulation, n_samples::Int, n::Int, missing_indices::Vector{Int} = Int[])
    NBMeanDispersionPopulationSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # γ (per observation)
        zeros(n_samples),           # r
        zeros(n_samples),           # logpost
        zeros(n_samples),           # α_ddcrp
        zeros(n_samples),           # s_ddcrp
        Matrix{Float64}(undef, n_samples, length(missing_indices)),  # y_imp
        missing_indices,            # missing_indices
    )
end

"""
    extract_samples!(model::NBMeanDispersionPopulation, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBMeanDispersionPopulation,
    state::NBMeanDispersionPopulationState,
    samples::NBMeanDispersionPopulationSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    samples.r[iter]    = state.r
    samples.γ[iter, :] = m_dict_to_samples(1:length(state.c), state.γ_dict)
end
