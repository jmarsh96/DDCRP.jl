# ============================================================================
# NBPopulationRates - NB with population offsets, augmented Gamma-Poisson
# ============================================================================
#
# Model (augmented representation):
#   y_i | λ_i, P_i    ~ Poisson(P_i · λ_i)
#   λ_i | γ_k, r      ~ Gamma(r, rate = r/γ_k)   [E[λ_i | γ_k] = γ_k]
#   γ_k               ~ InverseGamma(γ_a, γ_b)
#   r                 ~ Gamma(r_a, r_b)
#
# Marginalising λ_i recovers y_i | γ_k ~ NegBin(r, P_i·γ_k).
#
# Conjugate updates:
#   λ_i | y_i, γ_k, r ~ Gamma(y_i + r,  P_i + r/γ_k)            [exact Gibbs]
#   γ_k | {λ_i}_{i∈k} ~ InverseGamma(n_k·r + γ_a,  r·Λ_k + γ_b) [exact Gibbs]
#   r                  ~ MH random walk
#
# Table contribution (conditional on λ_i and γ_k):
#   TC_k = Σ_{i∈k} [ y_i·log(P_i) − log Γ(y_i+1) ]
#          + Σ_{i∈k} [ (y_i + r − 1)·log(λ_i) − P_i·λ_i ]
#          + n_k·(r·log(r) − log Γ(r))
#          + γ_a·log(γ_b) − log Γ(γ_a)
#          − (n_k·r + γ_a + 1)·log(γ_k) − (r·Λ_k + γ_b)/γ_k
#   where Λ_k = Σ_{i∈k} λ_i
#
# RJMCMC birth proposal for γ_k: conjugate posterior
#   InverseGamma(|S_i|·r + γ_a,  r·Λ_{S_i} + γ_b)
#
# Parameters: c, λ (individual rates), γ_k (cluster rates), r
# Requires: exposure/population data P_i via CountDataWithPopulation
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBPopulationRates <: NegativeBinomialModel

Negative Binomial model with population offsets using an augmented Gamma-Poisson
hierarchy. Observations follow y_i | λ_i ~ Poisson(P_i · λ_i), where individual
rates λ_i | γ_k, r ~ Gamma(r, r/γ_k) and cluster rates γ_k ~ InverseGamma(γ_a, γ_b).

Both λ_i and γ_k are maintained explicitly. λ_i and γ_k are updated via exact
conjugate Gibbs; customer assignments via RJMCMC; r via MH.

Requires exposure data P_i via CountDataWithPopulation.
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
- `λ::Vector{T}`: Individual-level latent rates
- `γ_dict::Dict{Vector{Int}, T}`: Table -> cluster rate mapping
- `r::T`: Global dispersion parameter
"""
mutable struct NBPopulationRatesState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ::Vector{T}
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
- `γ_a::T`: InverseGamma shape parameter for cluster rates γ_k
- `γ_b::T`: InverseGamma scale parameter for cluster rates γ_k
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NBPopulationRatesPriors{T<:Real} <: AbstractPriors
    γ_a::T
    γ_b::T
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
- `c::Matrix{Int}`: Customer assignments (n_samples × n_obs)
- `λ::Matrix{T}`: Individual-level rates (n_samples × n_obs)
- `γ::Matrix{T}`: Cluster rates per observation (n_samples × n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Matrix{T}
    γ::Matrix{T}
    r::Vector{T}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
    y_imp::Matrix{Float64}
    missing_indices::Vector{Int}
end

requires_population(::NBPopulationRates) = true

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::NBPopulationRatesState) = (γ = state.γ_dict,)

"""
    fixed_dim_params(model::NBPopulationRates, S_i, table_old, table_new, state, data, priors, opts)

Fixed-dimensional parameter update when moving set S_i transfers between existing clusters.
Uses NoUpdate strategy: cluster rates are kept at current values (corrected by the
next conjugate Gibbs step in update_params!).
"""
function fixed_dim_params(
    ::NBPopulationRates,
    S_i::Vector{Int},
    table_old::Vector{Int},
    table_new::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors,
    opts::MCMCOptions
)
    γ_depleted  = state.γ_dict[table_old]
    γ_augmented = state.γ_dict[table_new]
    return (γ = γ_depleted,), (γ = γ_augmented,), 0.0
end

"""
    sample_birth_params(model::NBPopulationRates, ::PriorProposal, S_i, state, data, priors)

Sample a new cluster rate γ for a birth move using the conjugate posterior
InverseGamma(|S_i|·r + γ_a, r·Λ_{S_i} + γ_b) conditioned on the current λ values
for the moving set S_i.
"""
function sample_birth_params(
    ::NBPopulationRates,
    ::PriorProposal,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    r    = state.r
    n_Si = length(S_i)
    Λ_Si = sum(state.λ[i] for i in S_i)

    α_post = n_Si * r + priors.γ_a
    β_post = r * Λ_Si + priors.γ_b
    Q      = InverseGamma(α_post, β_post)
    γ_new  = rand(Q)
    return (γ = γ_new,), logpdf(Q, γ_new)
end

"""
    birth_params_logpdf(model::NBPopulationRates, ::PriorProposal, params_old, S_i, state, data, priors)

Log-density of the birth proposal for a death (reverse) move.
Evaluates the conjugate posterior InverseGamma at the existing γ value.
"""
function birth_params_logpdf(
    ::NBPopulationRates,
    ::PriorProposal,
    params_old::NamedTuple,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    r    = state.r
    n_Si = length(S_i)
    Λ_Si = sum(state.λ[i] for i in S_i)

    α_post = n_Si * r + priors.γ_a
    β_post = r * Λ_Si + priors.γ_b
    Q      = InverseGamma(α_post, β_post)
    return logpdf(Q, params_old.γ)
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBPopulationRates, table, state, data, priors)

Compute log-contribution of a table conditional on explicit λ_i and γ_k.

TC_k = Σ_{i∈k} [ y_i·log(P_i) − log Γ(y_i+1) ]
     + Σ_{i∈k} [ (y_i + r − 1)·log(λ_i) − P_i·λ_i ]
     + n_k·(r·log(r) − log Γ(r))
     + γ_a·log(γ_b) − log Γ(γ_a)
     − (n_k·r + γ_a + 1)·log(γ_k) − (r·Λ_k + γ_b)/γ_k
"""
function table_contribution(
    ::NBPopulationRates,
    table::AbstractVector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    y  = observations(data)
    P  = population(data)
    γ  = state.γ_dict[sort(table)]
    r  = state.r
    n_k = length(table)

    y_k = view(y, table)
    P_k = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))
    λ_k = view(state.λ, table)
    Λ_k = sum(λ_k)

    # For missing y[j], treat as 0 (excludes Poisson likelihood contribution,
    # keeps only Gamma prior term (r-1)*log(λ_j) - P_j*λ_j for that obs)
    y_eff = any(ismissing, y) ? [ismissing(y_k[j]) ? 0.0 : Float64(y_k[j]) for j in 1:n_k] : Float64.(y_k)

    poisson_const = sum(y_eff[j] * log(P_k[j]) - loggamma(y_eff[j] + 1) for j in 1:n_k)
    lambda_terms  = sum((y_eff[j] + r - 1) * log(λ_k[j]) - P_k[j] * λ_k[j] for j in 1:n_k)
    gamma_norm    = n_k * (r * log(r) - loggamma(r))
    ig_norm       = priors.γ_a * log(priors.γ_b) - loggamma(priors.γ_a)
    gamma_kernel  = -(n_k * r + priors.γ_a + 1) * log(γ) - (r * Λ_k + priors.γ_b) / γ

    return poisson_const + lambda_terms + gamma_norm + ig_norm + gamma_kernel
end

"""
    impute_y(::NBPopulationRates, i, state, data, priors)

Draw a value for missing observation i from Poisson(P_i * λ_i) using the
current individual rate.
"""
function impute_y(
    ::NBPopulationRates,
    i::Int,
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    P = population(data)
    P_i = P isa Int ? Float64(P) : Float64(P[i])
    return rand(Poisson(P_i * state.λ[i]))
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBPopulationRates, data, state, priors, log_DDCRP)

Compute full log-posterior for the augmented NB population rates model.
"""
function posterior(
    model::NBPopulationRates,
    data::CountDataWithPopulation,
    state::NBPopulationRatesState,
    priors::NBPopulationRatesPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.γ_dict)) +
           logpdf(Gamma(priors.r_a, 1 / priors.r_b), state.r) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_λ_gibbs!(model::NBPopulationRates, state, data, priors, tables)

Update individual-level rates λ_i using exact conjugate Gibbs sampling.
Posterior: Gamma(y_i + r,  P_i + r/γ_k) for observation i in cluster k.
"""
function update_λ_gibbs!(
    ::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}}
)
    y = observations(data)
    P = population(data)
    r = state.r
    for table in tables
        γ = state.γ_dict[sort(table)]
        for i in table
            P_i    = P isa Int ? Float64(P) : Float64(P[i])
            α_post = Float64(y[i]) + r
            β_post = P_i + r / γ
            state.λ[i] = rand(Gamma(α_post, 1.0 / β_post))
        end
    end
end

"""
    update_γ_ig_rates!(model::NBPopulationRates, state, data, priors, tables)

Update cluster rates γ_k using exact conjugate Gibbs sampling.
Posterior: InverseGamma(n_k·r + γ_a,  r·Λ_k + γ_b) where Λ_k = Σ_{i∈k} λ_i.
"""
function update_γ_ig_rates!(
    ::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}}
)
    r = state.r
    for table in tables
        key = sort(table)
        n_k  = length(table)
        Λ_k  = sum(state.λ[i] for i in table)
        α_post = n_k * r + priors.γ_a
        β_post = r * Λ_k + priors.γ_b
        state.γ_dict[key] = rand(InverseGamma(α_post, β_post))
    end
end

"""
    update_r!(model::NBPopulationRates, state, data, priors, tables; prop_sd=0.5)

Update global dispersion r using Metropolis-Hastings with a normal random walk.
"""
function update_r!(
    model::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    r_can = rand(Normal(state.r, prop_sd))
    r_can <= 0 && return

    state_can = NBPopulationRatesState(state.c, state.λ, state.γ_dict, r_can)

    logpost_current   = sum(table_contribution(model, table, state, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1 / priors.r_b), state.r)
    logpost_candidate = sum(table_contribution(model, table, state_can, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1 / priors.r_b), r_can)

    if log(rand()) < logpost_candidate - logpost_current
        state.r = r_can
    end
end

"""
    update_params!(model::NBPopulationRates, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters: λ_i and γ_k via exact Gibbs, r via MH.
Assignment updates are handled by update_c!.
"""
function update_params!(
    model::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}},
    ::AbstractMatrix,
    opts::MCMCOptions
)
    if should_infer(opts, :λ)
        update_λ_gibbs!(model, state, data, priors, tables)
    end
    if should_infer(opts, :γ)
        update_γ_ig_rates!(model, state, data, priors, tables)
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

Create initial MCMC state. Assignments drawn from ddCRP prior; λ_i initialised
from smoothed empirical rates; γ_k drawn from conjugate posterior given initial λ;
r initialised to 1.0.
"""
function initialise_state(
    ::NBPopulationRates,
    data::CountDataWithPopulation,
    ddcrp_params::DDCRPParams,
    priors::NBPopulationRatesPriors
)
    y  = observations(data)
    P  = population(data)
    D  = distance_matrix(data)
    n  = length(y)
    r0 = 1.0

    c      = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)

    # Initialise λ_i from smoothed empirical rate
    P_vec = P isa Int ? fill(Float64(P), n) : Float64.(P)
    λ0    = [max(ismissing(y[i]) ? 0.01 : Float64(y[i]) / P_vec[i], 0.01) for i in 1:n]

    # Initialise γ_k from conjugate posterior given λ0
    γ_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        key    = sort(table)
        n_k    = length(table)
        Λ_k    = sum(λ0[i] for i in table)
        α_post = n_k * r0 + priors.γ_a
        β_post = r0 * Λ_k + priors.γ_b
        γ_dict[key] = rand(InverseGamma(α_post, β_post))
    end

    return NBPopulationRatesState(c, λ0, γ_dict, r0)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBPopulationRates, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBPopulationRates, n_samples::Int, n::Int, missing_indices::Vector{Int} = Int[])
    NBPopulationRatesSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # λ
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
    extract_samples!(model::NBPopulationRates, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBPopulationRates,
    state::NBPopulationRatesState,
    samples::NBPopulationRatesSamples,
    iter::Int
)
    samples.c[iter, :]  = state.c
    samples.λ[iter, :]  = state.λ
    samples.r[iter]     = state.r
    for (table, γ_val) in state.γ_dict
        for i in table
            samples.γ[iter, i] = γ_val
        end
    end
end
