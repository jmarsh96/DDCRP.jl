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
end

# ============================================================================
# Trait Functions
# ============================================================================

has_latent_rates(::PoissonPopulationRates) = false
has_global_dispersion(::PoissonPopulationRates) = false
has_cluster_dispersion(::PoissonPopulationRates) = false
has_cluster_means(::PoissonPopulationRates) = false
has_cluster_rates(::PoissonPopulationRates) = true
is_marginalised(::PoissonPopulationRates) = false
requires_trials(::PoissonPopulationRates) = true  # Requires exposure P

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::PoissonPopulationRates, table, state, data, priors)

Compute log-contribution of a table with population-adjusted Poisson likelihood.

# Arguments
- `data`: CountDataWithTrials containing y (counts) and N (exposures/populations as P)
"""
function table_contribution(
    ::PoissonPopulationRates,
    table::AbstractVector{Int},
    state::PoissonPopulationRatesState,
    data::CountDataWithTrials,
    priors::PoissonPopulationRatesPriors
)
    y = observations(data)
    P = trials(data)
    ρ = state.ρ_dict[sort(table)]

    # Poisson log-likelihood: y_i * log(P_i * ρ) - P_i * ρ - log(y_i!)
    # = y_i * log(P_i) + y_i * log(ρ) - P_i * ρ - log(y_i!)
    S_k = sum(view(y, table))
    sum_P = sum(view(P, table))
    log_P_term = sum(y[i] * log(P[i]) for i in table if P[i] > 0)

    log_lik = log_P_term + S_k * log(ρ) - sum_P * ρ - sum(loggamma.(view(y, table) .+ 1))

    # Gamma prior on ρ
    log_prior = (priors.ρ_a - 1) * log(ρ) - priors.ρ_b * ρ

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
    data::CountDataWithTrials,
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
    data::CountDataWithTrials,
    priors::PoissonPopulationRatesPriors,
    tables::Vector{Vector{Int}}
)
    y = observations(data)
    P = trials(data)
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
    data::CountDataWithTrials,
    priors::PoissonPopulationRatesPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    diagnostics = Vector{Tuple{Symbol, Int, Int, Bool}}()

    update_cluster_rates!(model, state, data, priors, tables)

    # Note: Assignment updates for this model would need RJMCMC implementation
    # Currently only supports parameter updates

    return diagnostics
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
    data::CountDataWithTrials,
    ddcrp_params::DDCRPParams,
    priors::PoissonPopulationRatesPriors
)
    y = observations(data)
    P = trials(data)
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
        zeros(n_samples)            # logpost
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
