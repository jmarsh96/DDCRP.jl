# ============================================================================
# PoissonPopulationRates - Population-based Poisson with cluster rate multipliers
# ============================================================================
#
# Model:
#   y_i | ρ_k, P_i ~ Poisson(P_i * ρ_k)    for observation i in cluster k
#   ρ_k ~ Gamma(ρ_a, ρ_b)                  (cluster rate multiplier)
#
# This model is useful when observations have different "exposures" P_i
# (e.g., population sizes, time periods) and we want to estimate cluster-specific
# rates per unit exposure.
#
# Parameters: c (assignments), ρ_k (cluster rate multipliers)
# Data: y (counts), P (exposures/populations)
# ============================================================================

using Distributions, SpecialFunctions, Random

# ============================================================================
# Type Definition
# ============================================================================

"""
    PoissonPopulationRates <: PoissonModel

Poisson model with population/exposure adjustment.
Rate for observation i in cluster k is λ_i = P_i * ρ_k.

Parameters:
- c: Customer assignments
- ρ_k: Cluster rate multipliers (cluster-level)

Requires exposure data P_i for each observation.
"""
struct PoissonPopulationRates <: PoissonModel end

# ============================================================================
# State Type
# ============================================================================

"""
    PoissonPopulationRatesState{T<:Real} <: AbstractMCMCState{T}

State for PoissonPopulationRates model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `ρ_dict::Dict{Vector{Int}, T}`: Table -> cluster rate multiplier mapping
"""
mutable struct PoissonPopulationRatesState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    ρ_dict::Dict{Vector{Int}, T}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    PoissonPopulationRatesPriors{T<:Real} <: AbstractPriors

Prior specification for PoissonPopulationRates model.

# Fields
- `ρ_a::T`: Gamma shape parameter for rate multiplier ρ
- `ρ_b::T`: Gamma rate parameter for rate multiplier ρ
"""
struct PoissonPopulationRatesPriors{T<:Real} <: AbstractPriors
    ρ_a::T
    ρ_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    PoissonPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for PoissonPopulationRates model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `ρ::Matrix{T}`: Cluster rate multipliers per observation (n_samples x n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct PoissonPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    ρ::Matrix{T}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
end


requires_population(::PoissonPopulationRates) = true

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::PoissonPopulationRatesState) = (ρ = state.ρ_dict,)

# PriorProposal samples from the conjugate posterior: Gamma(ρ_a + S_k, 1/(ρ_b + sum_P_k))
function sample_birth_params(::PoissonPopulationRates, ::PriorProposal,
                             S_i::Vector{Int}, state::PoissonPopulationRatesState,
                             data::CountDataWithPopulation, priors::PoissonPopulationRatesPriors)
    y = observations(data)
    P = population(data)
    S_k   = sum(view(y, S_i))
    sum_P = sum(view(P, S_i))
    Q = Gamma(priors.ρ_a + S_k, 1.0 / (priors.ρ_b + sum_P))
    ρ_new = rand(Q)
    return (ρ = ρ_new,), logpdf(Q, ρ_new)
end

function birth_params_logpdf(::PoissonPopulationRates, ::PriorProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::PoissonPopulationRatesState, data::CountDataWithPopulation,
                             priors::PoissonPopulationRatesPriors)
    y = observations(data)
    P = population(data)
    S_k   = sum(view(y, S_i))
    sum_P = sum(view(P, S_i))
    Q = Gamma(priors.ρ_a + S_k, 1.0 / (priors.ρ_b + sum_P))
    return logpdf(Q, params_old.ρ)
end

# NormalMomentMatch — log-Normal proposal centred at the MLE log(S_k / sum_P).
# Uses the moving set's own sufficient statistics so the proposal is symmetric
# across birth and death moves (no state look-up required).
# Falls back to the prior mean when S_k = 0.
function sample_birth_params(::PoissonPopulationRates, prop::NormalMomentMatch,
                             S_i::Vector{Int}, state::PoissonPopulationRatesState,
                             data::CountDataWithPopulation, priors::PoissonPopulationRatesPriors)
    y = observations(data)
    P = population(data)
    S_k   = sum(view(y, S_i))
    sum_P = sum(view(P, S_i))
    ρ_mle = S_k > 0 ? S_k / sum_P : priors.ρ_a / priors.ρ_b
    Q = Normal(log(ρ_mle), prop.σ[1])
    log_ρ_new = rand(Q)
    ρ_new = exp(log_ρ_new)
    # Jacobian: d(log ρ)/dρ = 1/ρ  →  p(ρ) = p(log ρ) / ρ
    log_q = logpdf(Q, log_ρ_new) - log_ρ_new
    return (ρ = ρ_new,), log_q
end

function birth_params_logpdf(::PoissonPopulationRates, prop::NormalMomentMatch,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::PoissonPopulationRatesState, data::CountDataWithPopulation,
                             priors::PoissonPopulationRatesPriors)
    y = observations(data)
    P = population(data)
    S_k   = sum(view(y, S_i))
    sum_P = sum(view(P, S_i))
    ρ_mle = S_k > 0 ? S_k / sum_P : priors.ρ_a / priors.ρ_b
    Q = Normal(log(ρ_mle), prop.σ[1])
    log_ρ = log(params_old.ρ)
    return logpdf(Q, log_ρ) - log_ρ
end

# LogNormalMomentMatch — log-Normal proposal centred at the conjugate posterior
# mean log((ρ_a + S_k) / (ρ_b + sum_P)).  Approximates the conditional posterior
# (Gamma) with a log-Normal whose centre is data-driven; σ[1] is a tuning parameter.
function sample_birth_params(::PoissonPopulationRates, prop::LogNormalMomentMatch,
                             S_i::Vector{Int}, state::PoissonPopulationRatesState,
                             data::CountDataWithPopulation, priors::PoissonPopulationRatesPriors)
    y = observations(data)
    P = population(data)
    S_k   = sum(view(y, S_i))
    sum_P = sum(view(P, S_i))
    ρ_post_mean = (priors.ρ_a + S_k) / (priors.ρ_b + sum_P)
    Q = Normal(log(ρ_post_mean), prop.σ[1])
    log_ρ_new = rand(Q)
    ρ_new = exp(log_ρ_new)
    log_q = logpdf(Q, log_ρ_new) - log_ρ_new
    return (ρ = ρ_new,), log_q
end

function birth_params_logpdf(::PoissonPopulationRates, prop::LogNormalMomentMatch,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::PoissonPopulationRatesState, data::CountDataWithPopulation,
                             priors::PoissonPopulationRatesPriors)
    y = observations(data)
    P = population(data)
    S_k   = sum(view(y, S_i))
    sum_P = sum(view(P, S_i))
    ρ_post_mean = (priors.ρ_a + S_k) / (priors.ρ_b + sum_P)
    Q = Normal(log(ρ_post_mean), prop.σ[1])
    log_ρ = log(params_old.ρ)
    return logpdf(Q, log_ρ) - log_ρ
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::PoissonPopulationRates, table, state, data, priors)

Compute log-contribution of a table with population-adjusted Poisson likelihood.

# Arguments
- `data`: CountDataWithPopulation containing y (counts) and N (exposures/populations as P)
"""
function table_contribution(
    ::PoissonPopulationRates,
    table::AbstractVector{Int},
    state::PoissonPopulationRatesState,
    data::CountDataWithPopulation,
    priors::PoissonPopulationRatesPriors
)
    y = observations(data)
    P = population(data)
    ρ = state.ρ_dict[sort(table)]

    # Poisson log-likelihood: y_i * log(P_i * ρ) - P_i * ρ - log(y_i!)
    # = y_i * log(P_i) + y_i * log(ρ) - P_i * ρ - log(y_i!)
    S_k = sum(view(y, table))
    sum_P = sum(view(P, table))
    log_P_term = sum(y[i] * log(P[i]) for i in table if P[i] > 0)

    log_lik = log_P_term + S_k * log(ρ) - sum_P * ρ - sum(loggamma.(view(y, table) .+ 1))

    # Complete Gamma prior on ρ (normalising constant must be included: it appears
    # once per cluster, so it does not cancel in birth/death acceptance ratios)
    log_prior = (priors.ρ_a - 1) * log(ρ) - priors.ρ_b * ρ +
                priors.ρ_a * log(priors.ρ_b) - loggamma(priors.ρ_a)

    return log_lik + log_prior
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::PoissonPopulationRates, data, state, priors, log_DDCRP)

Compute full log-posterior for population-adjusted Poisson model.
"""
function posterior(
    model::PoissonPopulationRates,
    data::CountDataWithPopulation,
    state::PoissonPopulationRatesState,
    priors::PoissonPopulationRatesPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.ρ_dict)) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_cluster_rates!(model::PoissonPopulationRates, state, data, priors, tables)

Update cluster rate multipliers using conjugate Gibbs sampling.
Posterior: Gamma(ρ_a + S_k, ρ_b + sum_P_k)
"""
function update_cluster_rates!(
    ::PoissonPopulationRates,
    state::PoissonPopulationRatesState,
    data::CountDataWithPopulation,
    priors::PoissonPopulationRatesPriors,
    tables::Vector{Vector{Int}}
)
    y = observations(data)
    P = population(data)
    for table in tables
        key = sort(table)
        S_k = sum(view(y, table))
        sum_P = sum(view(P, table))

        # Conjugate posterior: Gamma(α + S_k, β + sum_P)
        α_post = priors.ρ_a + S_k
        β_post = priors.ρ_b + sum_P

        state.ρ_dict[key] = rand(Gamma(α_post, 1/β_post))
    end
end

"""
    update_params!(model::PoissonPopulationRates, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters.
"""
function update_params!(
    model::PoissonPopulationRates,
    state::PoissonPopulationRatesState,
    data::CountDataWithPopulation,
    priors::PoissonPopulationRatesPriors,
    tables::Vector{Vector{Int}},
    ::AbstractMatrix,
    ::MCMCOptions
)
    update_cluster_rates!(model, state, data, priors, tables)
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::PoissonPopulationRates, data, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::PoissonPopulationRates,
    data::CountDataWithPopulation,
    ddcrp_params::DDCRPParams,
    priors::PoissonPopulationRatesPriors
)
    y = observations(data)
    P = population(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)

    ρ_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        # Initialize at empirical rate
        S_k = sum(view(y, table))
        sum_P = sum(view(P, table))
        ρ_dict[sort(table)] = sum_P > 0 ? S_k / sum_P : 1.0
    end

    return PoissonPopulationRatesState(c, ρ_dict)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::PoissonPopulationRates, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::PoissonPopulationRates, n_samples::Int, n::Int)
    PoissonPopulationRatesSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # ρ - stores cluster ρ per observation
        zeros(n_samples),           # logpost
        zeros(n_samples),           # α_ddcrp
        zeros(n_samples),           # s_ddcrp
    )
end

"""
    extract_samples!(model::PoissonPopulationRates, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::PoissonPopulationRates,
    state::PoissonPopulationRatesState,
    samples::PoissonPopulationRatesSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    for (table, ρ_val) in state.ρ_dict
        for i in table
            samples.ρ[iter, i] = ρ_val
        end
    end
end

# ============================================================================
# Per-parameter dispatch — required by Resample fixed-dim proposal
# ============================================================================
# sample_birth_param / birth_param_logpdf with Val{:ρ} delegate to the
# corresponding plural versions, enabling Resample(proposal) as a
# fixed-dimension proposal for any supported birth proposal.

function sample_birth_param(
    model::PoissonPopulationRates,
    ::Val{:ρ},
    proposal::BirthProposal,
    S_i::Vector{Int},
    state::PoissonPopulationRatesState,
    data::CountDataWithPopulation,
    priors::PoissonPopulationRatesPriors
)
    params, lq = sample_birth_params(model, proposal, S_i, state, data, priors)
    return params.ρ, lq
end

function birth_param_logpdf(
    model::PoissonPopulationRates,
    ::Val{:ρ},
    proposal::BirthProposal,
    ρ_val,
    S_i::Vector{Int},
    state::PoissonPopulationRatesState,
    data::CountDataWithPopulation,
    priors::PoissonPopulationRatesPriors
)
    return birth_params_logpdf(model, proposal, (ρ = ρ_val,), S_i, state, data, priors)
end
