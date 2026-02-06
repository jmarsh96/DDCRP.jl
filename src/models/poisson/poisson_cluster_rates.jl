# ============================================================================
# PoissonClusterRates - Poisson with explicit cluster rates
# ============================================================================
#
# Model:
#   y_i | λ_k ~ Poisson(λ_k)    for observation i in cluster k
#   λ_k ~ Gamma(λ_a, λ_b)       (explicit, sampled via conjugacy)
#
# Parameters: c (assignments), λ_k (cluster rates)
# ============================================================================

using Distributions, SpecialFunctions, Random

# ============================================================================
# Type Definition
# ============================================================================

"""
    PoissonClusterRates <: PoissonModel

Poisson model with explicit cluster-specific rates.
Rates λ_k are maintained and updated via conjugate Gibbs sampling.

Parameters:
- c: Customer assignments
- λ_k: Cluster rates (cluster-level)
"""
struct PoissonClusterRates <: PoissonModel end

# ============================================================================
# State Type
# ============================================================================

"""
    PoissonClusterRatesState{T<:Real} <: AbstractMCMCState{T}

State for PoissonClusterRates model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `λ_dict::Dict{Vector{Int}, T}`: Table -> cluster rate mapping
"""
mutable struct PoissonClusterRatesState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ_dict::Dict{Vector{Int}, T}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    PoissonClusterRatesPriors{T<:Real} <: AbstractPriors

Prior specification for PoissonClusterRates model.

# Fields
- `λ_a::T`: Gamma shape parameter for rate λ
- `λ_b::T`: Gamma rate parameter for rate λ
"""
struct PoissonClusterRatesPriors{T<:Real} <: AbstractPriors
    λ_a::T
    λ_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    PoissonClusterRatesSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for PoissonClusterRates model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `λ::Matrix{T}`: Cluster rates per observation (n_samples x n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct PoissonClusterRatesSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Matrix{T}
    logpost::Vector{T}
end

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::PoissonClusterRatesState) = (λ = state.λ_dict,)
copy_cluster_param_dicts(state::PoissonClusterRatesState) = (λ = copy(state.λ_dict),)

function make_candidate_state(::PoissonClusterRates, state::PoissonClusterRatesState,
                              c_can::Vector{Int}, params_can::NamedTuple)
    PoissonClusterRatesState(c_can, params_can.λ)
end

function commit_params!(state::PoissonClusterRatesState, params_can::NamedTuple)
    empty!(state.λ_dict); merge!(state.λ_dict, params_can.λ)
end

# --- PriorProposal (samples from conjugate posterior) ---
function sample_birth_params(::PoissonClusterRates, ::PriorProposal,
                             S_i::Vector{Int}, state::PoissonClusterRatesState,
                             data::CountData, priors::PoissonClusterRatesPriors)
    y = observations(data)
    S_k = sum(view(y, S_i))
    n_k = length(S_i)
    Q = Gamma(priors.λ_a + S_k, 1/(priors.λ_b + n_k))
    λ_new = rand(Q)
    return (λ = λ_new,), logpdf(Q, λ_new)
end

function birth_params_logpdf(::PoissonClusterRates, ::PriorProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::PoissonClusterRatesState, data::CountData,
                             priors::PoissonClusterRatesPriors)
    y = observations(data)
    S_k = sum(view(y, S_i))
    n_k = length(S_i)
    Q = Gamma(priors.λ_a + S_k, 1/(priors.λ_b + n_k))
    return logpdf(Q, params_old.λ)
end

# --- FixedDistributionProposal ---
function sample_birth_params(::PoissonClusterRates, prop::FixedDistributionProposal,
                             S_i::Vector{Int}, state::PoissonClusterRatesState,
                             data::CountData, priors::PoissonClusterRatesPriors)
    Q = prop.dists[1]
    λ_new = rand(Q)
    return (λ = λ_new,), logpdf(Q, λ_new)
end

function birth_params_logpdf(::PoissonClusterRates, prop::FixedDistributionProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::PoissonClusterRatesState, data::CountData,
                             priors::PoissonClusterRatesPriors)
    return logpdf(prop.dists[1], params_old.λ)
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::PoissonClusterRates, table, state, data, priors)

Compute log-contribution of a table with explicit cluster rate.
"""
function table_contribution(
    ::PoissonClusterRates,
    table::AbstractVector{Int},
    state::PoissonClusterRatesState,
    data::CountData,
    priors::PoissonClusterRatesPriors
)
    y = observations(data)
    λ = state.λ_dict[sort(table)]
    n_k = length(table)
    S_k = sum(view(y, table))

    # Poisson likelihood + Gamma prior on λ
    log_lik = S_k * log(λ) - n_k * λ - sum(loggamma.(view(y, table) .+ 1))
    log_prior = (priors.λ_a - 1) * log(λ) - priors.λ_b * λ

    return log_lik + log_prior
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::PoissonClusterRates, data, state, priors, log_DDCRP)

Compute full log-posterior for Poisson model with explicit rates.
"""
function posterior(
    model::PoissonClusterRates,
    data::CountData,
    state::PoissonClusterRatesState,
    priors::PoissonClusterRatesPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.λ_dict)) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_cluster_rates!(model::PoissonClusterRates, state, data, priors, tables)

Update cluster rates using conjugate Gibbs sampling.
Posterior: Gamma(λ_a + S_k, λ_b + n_k)
"""
function update_cluster_rates!(
    ::PoissonClusterRates,
    state::PoissonClusterRatesState,
    data::CountData,
    priors::PoissonClusterRatesPriors,
    tables::Vector{Vector{Int}}
)
    y = observations(data)
    for table in tables
        key = sort(table)
        n_k = length(table)
        S_k = sum(view(y, table))

        # Conjugate posterior: Gamma(α + S_k, β + n_k)
        α_post = priors.λ_a + S_k
        β_post = priors.λ_b + n_k

        state.λ_dict[key] = rand(Gamma(α_post, 1/β_post))
    end
end

"""
    update_params!(model::PoissonClusterRates, state, data, priors, tables, log_DDCRP, opts)

Update cluster rates via conjugate Gibbs sampling.
Assignment updates are handled separately by `update_c!`.
"""
function update_params!(
    model::PoissonClusterRates,
    state::PoissonClusterRatesState,
    data::CountData,
    priors::PoissonClusterRatesPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    update_cluster_rates!(model, state, data, priors, tables)
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::PoissonClusterRates, data, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::PoissonClusterRates,
    data::CountData,
    ddcrp_params::DDCRPParams,
    priors::PoissonClusterRatesPriors
)
    y = observations(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)

    λ_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        # Initialize at posterior mean
        S_k = sum(view(y, table))
        n_k = length(table)
        λ_dict[sort(table)] = (priors.λ_a + S_k) / (priors.λ_b + n_k)
    end

    return PoissonClusterRatesState(c, λ_dict)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::PoissonClusterRates, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::PoissonClusterRates, n_samples::Int, n::Int)
    PoissonClusterRatesSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # λ - stores cluster rate per observation
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::PoissonClusterRates, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::PoissonClusterRates,
    state::PoissonClusterRatesState,
    samples::PoissonClusterRatesSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    # Store λ per observation (each obs gets its cluster's rate)
    for (table, λ_val) in state.λ_dict
        for i in table
            samples.λ[iter, i] = λ_val
        end
    end
end
