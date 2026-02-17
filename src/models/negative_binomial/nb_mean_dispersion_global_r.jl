# ============================================================================
# NBMeanDispersionGlobalR - Direct NegBin with cluster means and global r
# ============================================================================
#
# Model:
#   y_i | m_k, r ~ NegBin(r, r/(r + m_k))  for observation i in cluster k
#   m_k ~ InverseGamma(m_a, m_b)           (explicit, sampled)
#   r ~ Gamma(r_a, r_b)                    (global dispersion)
#
# This is the mean-dispersion parameterisation of NegBin where:
#   E[y] = m_k
#   Var[y] = m_k + m_k²/r
#
# Parameters: c (assignments), m_k (cluster means), r (global dispersion)
# No latent λ rates - direct NegBin likelihood
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBMeanDispersionGlobalR <: NegativeBinomialModel

Negative Binomial model using direct mean-dispersion parameterisation.
No latent rates - uses NegBin(m_k, r) likelihood directly.

Parameters:
- m_k: Cluster means (cluster-level)
- c: Customer assignments
- r: Global dispersion parameter
"""
struct NBMeanDispersionGlobalR <: NegativeBinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    NBMeanDispersionGlobalRState{T<:Real} <: AbstractMCMCState{T}

State for NBMeanDispersionGlobalR model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `m_dict::Dict{Vector{Int}, T}`: Table -> cluster mean mapping
- `r::T`: Global dispersion parameter
"""
mutable struct NBMeanDispersionGlobalRState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    m_dict::Dict{Vector{Int}, T}
    r::T
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBMeanDispersionGlobalRPriors{T<:Real} <: AbstractPriors

Prior specification for NBMeanDispersionGlobalR model.

# Fields
- `m_a::T`: InverseGamma shape parameter for cluster mean m
- `m_b::T`: InverseGamma scale parameter for cluster mean m
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NBMeanDispersionGlobalRPriors{T<:Real} <: AbstractPriors
    m_a::T
    m_b::T
    r_a::T
    r_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    NBMeanDispersionGlobalRSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for NBMeanDispersionGlobalR model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `m::Matrix{T}`: Cluster means per observation (n_samples x n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBMeanDispersionGlobalRSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    r::Vector{T}
    m::Matrix{T}
    logpost::Vector{T}
end

# ============================================================================
# Trait Functions
# ============================================================================

is_marginalised(::NBMeanDispersionGlobalR) = false

# ============================================================================
# Table Contribution
# ============================================================================

# Note: negbin_logpdf is defined in core/state.jl

"""
    table_contribution(model::NBMeanDispersionGlobalR, table, state, data, priors)

Compute log-contribution of a table with direct NegBin likelihood.
"""
function table_contribution(
    ::NBMeanDispersionGlobalR,
    table::AbstractVector{Int},
    state::NBMeanDispersionGlobalRState,
    data::CountData,
    priors::NBMeanDispersionGlobalRPriors
)
    y = observations(data)
    m = state.m_dict[sort(table)]
    r = state.r

    # NegBin log-likelihood for all observations in table
    log_lik = sum(negbin_logpdf(y[i], m, r) for i in table)

    # InverseGamma prior on m
    log_prior_m = logpdf(InverseGamma(priors.m_a, priors.m_b), m)

    return log_lik + log_prior_m
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBMeanDispersionGlobalR, data, state, priors, log_DDCRP)

Compute full log-posterior for direct NegBin model.
"""
function posterior(
    model::NBMeanDispersionGlobalR,
    data::CountData,
    state::NBMeanDispersionGlobalRState,
    priors::NBMeanDispersionGlobalRPriors,
    log_DDCRP::AbstractMatrix
)
    # Prior on r
    log_prior_r = logpdf(Gamma(priors.r_a, 1/priors.r_b), state.r)

    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.m_dict)) +
           ddcrp_contribution(state.c, log_DDCRP) +
           log_prior_r
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_m!(model::NBMeanDispersionGlobalR, state, data, priors; prop_sd=0.5)

Update all cluster means using Metropolis-Hastings.
"""
function update_m!(
    model::NBMeanDispersionGlobalR,
    state::NBMeanDispersionGlobalRState,
    data::CountData,
    priors::NBMeanDispersionGlobalRPriors;
    prop_sd::Float64 = 0.5
)
    for table in keys(state.m_dict)
        update_m_table!(model, table, state, data, priors; prop_sd=prop_sd)
    end
end

"""
    update_m_table!(model::NBMeanDispersionGlobalR, table, state, data, priors; prop_sd=0.5)

Update cluster mean for a single table.
"""
function update_m_table!(
    model::NBMeanDispersionGlobalR,
    table::Vector{Int},
    state::NBMeanDispersionGlobalRState,
    data::CountData,
    priors::NBMeanDispersionGlobalRPriors;
    prop_sd::Float64 = 0.5
)
    m_can = copy(state.m_dict)
    m_can[table] = rand(Normal(state.m_dict[table], prop_sd))

    m_can[table] <= 0 && return

    state_can = NBMeanDispersionGlobalRState(state.c, m_can, state.r)

    logpost_current = table_contribution(model, table, state, data, priors)
    logpost_candidate = table_contribution(model, table, state_can, data, priors)

    log_accept_ratio = logpost_candidate - logpost_current

    if log(rand()) < log_accept_ratio
        state.m_dict[table] = m_can[table]
    end
end

"""
    update_r!(model::NBMeanDispersionGlobalR, state, data, priors, tables; prop_sd=0.5)

Update global dispersion parameter r using Metropolis-Hastings.
"""
function update_r!(
    model::NBMeanDispersionGlobalR,
    state::NBMeanDispersionGlobalRState,
    data::CountData,
    priors::NBMeanDispersionGlobalRPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    r_can = rand(Normal(state.r, prop_sd))
    r_can <= 0 && return

    state_can = NBMeanDispersionGlobalRState(state.c, state.m_dict, r_can)

    logpost_current = sum(table_contribution(model, table, state, data, priors) for table in tables) +
                      logpdf(Gamma(priors.r_a, 1/priors.r_b), state.r)
    logpost_candidate = sum(table_contribution(model, table, state_can, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1/priors.r_b), r_can)

    log_accept_ratio = logpost_candidate - logpost_current

    if log(rand()) < log_accept_ratio
        state.r = r_can
    end
end

"""
    update_params!(model::NBMeanDispersionGlobalR, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters (m_k and r).
"""
function update_params!(
    model::NBMeanDispersionGlobalR,
    state::NBMeanDispersionGlobalRState,
    data::CountData,
    priors::NBMeanDispersionGlobalRPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    if should_infer(opts, :m)
        update_m!(model, state, data, priors; prop_sd=get_prop_sd(opts, :m))
    end

    if should_infer(opts, :r)
        update_r!(model, state, data, priors, tables; prop_sd=get_prop_sd(opts, :r))
    end
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::NBMeanDispersionGlobalR, data, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::NBMeanDispersionGlobalR,
    data::CountData,
    ddcrp_params::DDCRPParams,
    priors::NBMeanDispersionGlobalRPriors
)
    y = observations(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)
    m_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        # Initialize m at empirical mean of y in cluster
        m_dict[sort(table)] = max(mean(view(y, table)), 0.1)
    end
    r = 1.0
    return NBMeanDispersionGlobalRState(c, m_dict, r)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBMeanDispersionGlobalR, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBMeanDispersionGlobalR, n_samples::Int, n::Int)
    NBMeanDispersionGlobalRSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples),           # r
        zeros(n_samples, n),        # m (per observation)
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::NBMeanDispersionGlobalR, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBMeanDispersionGlobalR,
    state::NBMeanDispersionGlobalRState,
    samples::NBMeanDispersionGlobalRSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    samples.r[iter] = state.r
    samples.m[iter, :] = m_dict_to_samples(1:length(state.c), state.m_dict)
end
