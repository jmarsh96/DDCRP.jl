# ============================================================================
# NBPopulationRates - NB with population offsets, explicit cluster rates
# ============================================================================
#
# Model:
#   y_i | γ_k, P_i   ~ Poisson(P_i · γ_k)
#   γ_k               ~ Gamma(r, rate = r/μ)   [E[γ_k] = μ]  (explicit, sampled)
#   r                 ~ Gamma(r_a, r_b)         (global dispersion)
#
# Conjugate updates:
#   γ_k | y, c, r ~ Gamma(r + S_k,  r/μ + P_sum)   [Gibbs, where S_k=Σy_i, P_sum=ΣP_i]
#   r              ~ MH random walk
#
# Table contribution (conditional on γ_k):
#   TC_k = Σ_{i∈k} [y_i·log(P_i) − log Γ(y_i+1)]
#          + r·log(r) − r·log(μ) − log Γ(r)
#          + (r − 1 + S_k)·log(γ_k) − (P_sum + r/μ)·γ_k
#
# Parameters: c, γ_k (cluster rates), r
# RJMCMC birth proposal: conjugate posterior Gamma(r+S_Si, r/μ+P_Si) for moving set
# Requires: exposure/population data P_i via CountDataWithTrials
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBPopulationRates <: NegativeBinomialModel

Negative Binomial model with population/exposure offsets and global dispersion r.
Observations follow y_i | γ_k ~ Poisson(P_i · γ_k). Cluster-specific rates γ_k
are explicitly maintained and updated via conjugate Gibbs sampling.
Customer assignments are updated via RJMCMC.

Parameters:
- c: Customer assignments
- γ_k: Cluster rates (cluster-level, updated via conjugate Gibbs)
- r: Global dispersion parameter (updated via MH)

Requires exposure data P_i via CountDataWithTrials.
"""
struct NBPopulationRates <: NegativeBinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    NBPopulationRatesState{T<:Real} <: AbstractMCMCState{T}

State for NBPopulationRates model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `γ_dict::Dict{Vector{Int}, T}`: Table -> cluster rate mapping
- `r::T`: Global dispersion parameter
"""
mutable struct NBPopulationRatesState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    γ_dict::Dict{Vector{Int}, T}
    r::T
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBPopulationRatesPriors{T<:Real} <: AbstractPriors

Prior specification for NBPopulationRates model.

# Fields
- `μ::T`: Prior mean for cluster rates γ_k (Gamma(r, r/μ) has mean μ)
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NBPopulationRatesPriors{T<:Real} <: AbstractPriors
    μ::T
    r_a::T
    r_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    NBPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for NBPopulationRates model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `γ::Matrix{T}`: Cluster rates per observation (n_samples x n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    γ::Matrix{T}
    r::Vector{T}
    logpost::Vector{T}
end

requires_trials(::NBPopulationRates) = true

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::NBPopulationRatesState) = (γ = state.γ_dict,)

"""
    fixed_dim_params(model::NBPopulationRates, S_i, table_old, table_new, state, data, priors, opts)

Fixed-dimensional parameter update when moving set S_i transfers between existing clusters.
Uses NoUpdate strategy: cluster rates are kept at current values.
The acceptance ratio therefore depends only on the change in table contributions.
"""
function fixed_dim_params(
    ::NBPopulationRates,
    S_i::Vector{Int},
    table_old::Vector{Int},
    table_new::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors,
    opts::MCMCOptions
)
    γ_depleted  = state.γ_dict[table_old]
    γ_augmented = state.γ_dict[table_new]
    return (γ = γ_depleted,), (γ = γ_augmented,), 0.0
end

"""
    sample_birth_params(model::NBPopulationRates, ::PriorProposal, S_i, state, data, priors)

Sample a new cluster rate γ for a birth move. Uses the conjugate posterior
Gamma(r + S_Si, r/μ + P_Si) conditioned on the moving set, where
S_Si = Σ_{i∈S_i} y_i and P_Si = Σ_{i∈S_i} P_i.
"""
function sample_birth_params(
    ::NBPopulationRates,
    ::PriorProposal,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors
)
    y = observations(data)
    P = trials(data)
    r = state.r
    n_Si = length(S_i)

    P_vec = P isa Int ? fill(Float64(P), n_Si) : Float64.(view(P, S_i))
    S_sum = Float64(sum(y[j] for j in S_i))
    P_sum = sum(P_vec)

    α_post = r + S_sum
    β_post = r / priors.μ + P_sum
    Q = Gamma(α_post, 1.0 / β_post)
    γ_new = rand(Q)
    return (γ = γ_new,), logpdf(Q, γ_new)
end

"""
    birth_params_logpdf(model::NBPopulationRates, ::PriorProposal, params_old, S_i, state, data, priors)

Log-density of the birth proposal for a death (reverse) move.
Evaluates the conjugate posterior Gamma at the existing γ value.
"""
function birth_params_logpdf(
    ::NBPopulationRates,
    ::PriorProposal,
    params_old::NamedTuple,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors
)
    y = observations(data)
    P = trials(data)
    r = state.r
    n_Si = length(S_i)

    P_vec = P isa Int ? fill(Float64(P), n_Si) : Float64.(view(P, S_i))
    S_sum = Float64(sum(y[j] for j in S_i))
    P_sum = sum(P_vec)

    α_post = r + S_sum
    β_post = r / priors.μ + P_sum
    Q = Gamma(α_post, 1.0 / β_post)
    return logpdf(Q, params_old.γ)
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBPopulationRates, table, state, data, priors)

Compute log-contribution of a table conditional on explicit cluster rate γ_k.
Includes the Poisson log-likelihood for y_i given γ_k and P_i, and the
Gamma(r, r/μ) log-prior for γ_k.
"""
function table_contribution(
    ::NBPopulationRates,
    table::AbstractVector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors
)
    y = observations(data)
    P = trials(data)
    γ = state.γ_dict[sort(table)]
    r = state.r
    n_k = length(table)

    y_k = view(y, table)
    P_k = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))

    S_k   = Float64(sum(y_k))
    P_sum = sum(P_k)

    data_term  = sum(y_k[j] * log(P_k[j]) - loggamma(Float64(y_k[j]) + 1) for j in 1:n_k)
    prior_term = r * log(r) - r * log(priors.μ) - loggamma(r)
    gamma_term = (r - 1 + S_k) * log(γ) - (P_sum + r / priors.μ) * γ

    return data_term + prior_term + gamma_term
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBPopulationRates, data, state, priors, log_DDCRP)

Compute full log-posterior for the non-marginalised NB population rates model.
"""
function posterior(
    model::NBPopulationRates,
    data::CountDataWithTrials,
    state::NBPopulationRatesState,
    priors::NBPopulationRatesPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.γ_dict)) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_cluster_gamma_rates!(model::NBPopulationRates, state, data, priors, tables)

Update cluster rates γ_k using conjugate Gibbs sampling.
Posterior: Gamma(r + S_k, r/μ + P_sum) where S_k=Σy_i, P_sum=ΣP_i for cluster k.
"""
function update_cluster_gamma_rates!(
    ::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}}
)
    y = observations(data)
    P = trials(data)
    r = state.r
    for table in tables
        key = sort(table)
        n_k = length(table)
        y_k = view(y, table)
        P_vec = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))

        S_k   = Float64(sum(y_k))
        P_sum = sum(P_vec)

        α_post = r + S_k
        β_post = r / priors.μ + P_sum
        state.γ_dict[key] = rand(Gamma(α_post, 1.0 / β_post))
    end
end

"""
    update_r!(model::NBPopulationRates, state, data, priors, tables; prop_sd=0.5)

Update global dispersion parameter r using Metropolis-Hastings.
"""
function update_r!(
    model::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    r_can = rand(Normal(state.r, prop_sd))
    r_can <= 0 && return

    state_can = NBPopulationRatesState(state.c, state.γ_dict, r_can)

    logpost_current = sum(table_contribution(model, table, state, data, priors) for table in tables) +
                      logpdf(Gamma(priors.r_a, 1 / priors.r_b), state.r)
    logpost_candidate = sum(table_contribution(model, table, state_can, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1 / priors.r_b), r_can)

    if log(rand()) < logpost_candidate - logpost_current
        state.r = r_can
    end
end

"""
    update_params!(model::NBPopulationRates, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters (γ_k via Gibbs, r via MH).
Assignment updates are handled by update_c!.
"""
function update_params!(
    model::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}},
    ::AbstractMatrix,
    opts::MCMCOptions
)
    if should_infer(opts, :γ)
        update_cluster_gamma_rates!(model, state, data, priors, tables)
    end

    if should_infer(opts, :r)
        update_r!(model, state, data, priors, tables; prop_sd=get_prop_sd(opts, :r))
    end
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::NBPopulationRates, data, ddcrp_params, priors)

Create initial MCMC state. Assignments are drawn from the ddCRP prior;
γ_k is initialised from the empirical rate sum(y_k)/sum(P_k) per cluster.
"""
function initialise_state(
    ::NBPopulationRates,
    data::CountDataWithTrials,
    ddcrp_params::DDCRPParams,
    priors::NBPopulationRatesPriors
)
    y = observations(data)
    P = trials(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)

    γ_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        key = sort(table)
        n_k = length(table)
        y_k = view(y, table)
        P_vec = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))
        sum_y = Float64(sum(y_k))
        sum_P = sum(P_vec)
        γ_dict[key] = sum_y > 0 && sum_P > 0 ? sum_y / sum_P : priors.μ
    end

    return NBPopulationRatesState(c, γ_dict, 1.0)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBPopulationRates, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBPopulationRates, n_samples::Int, n::Int)
    NBPopulationRatesSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # γ (per observation)
        zeros(n_samples),           # r
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::NBPopulationRates, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBPopulationRates,
    state::NBPopulationRatesState,
    samples::NBPopulationRatesSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    samples.r[iter] = state.r
    for (table, γ_val) in state.γ_dict
        for i in table
            samples.γ[iter, i] = γ_val
        end
    end
end
