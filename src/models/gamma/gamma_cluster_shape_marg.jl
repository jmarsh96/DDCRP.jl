# ============================================================================
# GammaClusterShapeMarg - Gamma model with cluster-specific shape parameters
# ============================================================================
#
# Model:
#   y_i | α_k, β_k ~ Gamma(α_k, β_k)     for observation i in cluster k
#   β_k | α_k ~ Gamma(β_a, β_b)          (marginalised out)
#   α_k ~ Gamma(α_a, α_b)                (explicit, sampled via MH)
#
# The rate parameter β_k is marginalised out using Gamma-Gamma conjugacy.
# Only shape parameters α_k are explicitly sampled.
#
# Marginal likelihood (integrating out β_k) for n observations in cluster k:
#   log p(y_1,...,y_n | α, β_a, β_b) =
#       (α - 1) * Σ log(y_i)
#       - n * loggamma(α)
#       + loggamma(n*α + β_a)
#       - loggamma(β_a)
#       + β_a * log(β_b)
#       - (n*α + β_a) * log(Σy_i + β_b)
#
# Parameters: c (assignments), α_k (cluster shape parameters)
# Marginalised: β_k (cluster rate parameters)
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    GammaClusterShapeMarg <: GammaModel

Gamma model with cluster-specific shape parameters (α_k).
Rate parameters (β_k) are marginalised out using Gamma-Gamma conjugacy.

Parameters:
- α_k: Cluster shape parameters (cluster-level, explicit)
- c: Customer assignments

Marginalised: β_k (cluster rate parameters integrated out analytically)
"""
struct GammaClusterShapeMarg <: GammaModel end

# ============================================================================
# State Type
# ============================================================================

"""
    GammaClusterShapeMargState{T<:Real} <: AbstractMCMCState{T}

State for GammaClusterShapeMarg model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `α_dict::Dict{Vector{Int}, T}`: Table -> cluster shape parameter mapping
"""
mutable struct GammaClusterShapeMargState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    α_dict::Dict{Vector{Int}, T}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    GammaClusterShapeMargPriors{T<:Real} <: AbstractPriors

Prior specification for GammaClusterShapeMarg model.

# Fields
- `α_a::T`: Gamma shape parameter for α prior (shape of shape)
- `α_b::T`: Gamma rate parameter for α prior
- `β_a::T`: Gamma shape parameter for β prior (used in marginal likelihood)
- `β_b::T`: Gamma rate parameter for β prior (used in marginal likelihood)
"""
struct GammaClusterShapeMargPriors{T<:Real} <: AbstractPriors
    α_a::T
    α_b::T
    β_a::T
    β_b::T
end

# Default constructor with keyword arguments
function GammaClusterShapeMargPriors(;
    α_a::Real = 2.0,
    α_b::Real = 1.0,
    β_a::Real = 2.0,
    β_b::Real = 1.0
)
    T = promote_type(typeof(α_a), typeof(α_b), typeof(β_a), typeof(β_b))
    GammaClusterShapeMargPriors{T}(T(α_a), T(α_b), T(β_a), T(β_b))
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    GammaClusterShapeMargSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for GammaClusterShapeMarg model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `α::Matrix{T}`: Shape per observation (n_samples x n_obs) - stores cluster α
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct GammaClusterShapeMargSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    α::Matrix{T}
    logpost::Vector{T}
end

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::GammaClusterShapeMargState) = (α = state.α_dict,)

# --- PriorProposal ---
function sample_birth_params(::GammaClusterShapeMarg, ::PriorProposal, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)
    Q = Gamma(priors.α_a, 1/priors.α_b)
    α_new = rand(Q)
    return (α = α_new,), logpdf(Q, α_new)
end

function birth_params_logpdf(::GammaClusterShapeMarg, ::PriorProposal, params_old::NamedTuple, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)
    return logpdf(Gamma(priors.α_a, 1/priors.α_b), params_old.α)
end

# --- NormalMomentMatch ---
function sample_birth_params(::GammaClusterShapeMarg, prop::NormalMomentMatch, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)
    y = observations(data)
    α_est = fit_gamma_shape_moments(view(y, S_i))
    if isnothing(α_est)
        α_est = priors.α_a / priors.α_b
    end
    Q = truncated(Normal(α_est, prop.σ[1]), 0.0, Inf)
    α_new = rand(Q)
    return (α = α_new,), logpdf(Q, α_new)
end

function birth_params_logpdf(::GammaClusterShapeMarg, prop::NormalMomentMatch, params_old::NamedTuple, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)
    y = observations(data)
    α_est = fit_gamma_shape_moments(view(y, S_i))
    if isnothing(α_est)
        α_est = priors.α_a / priors.α_b
    end
    Q = truncated(Normal(α_est, prop.σ[1]), 0.0, Inf)
    return logpdf(Q, params_old.α)
end

# --- InverseGammaMomentMatch (fallback to prior for shape params) ---
function sample_birth_params(::GammaClusterShapeMarg, prop::InverseGammaMomentMatch, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)
    return sample_birth_params(GammaClusterShapeMarg(), PriorProposal(), S_i, state, data, priors)
end

function birth_params_logpdf(::GammaClusterShapeMarg, prop::InverseGammaMomentMatch, params_old::NamedTuple, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)    
    return birth_params_logpdf(GammaClusterShapeMarg(), PriorProposal(), params_old, S_i, state, data, priors)
end

# --- LogNormalMomentMatch ---
function sample_birth_params(::GammaClusterShapeMarg, prop::LogNormalMomentMatch, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)
    y = observations(data)
    α_est = nothing
    if length(S_i) >= prop.min_size
        α_est = fit_gamma_shape_moments(view(y, S_i))
    end
    if isnothing(α_est)
        α_est = priors.α_a / priors.α_b
    end
    α_est = max(α_est, 0.01)
    log_α_est = log(α_est)
    log_α_new = rand(Normal(log_α_est, prop.σ[1]))
    α_new = exp(log_α_new)
    log_q = logpdf(Normal(log_α_est, prop.σ[1]), log_α_new) - log_α_new
    return (α = α_new,), log_q
end

function birth_params_logpdf(::GammaClusterShapeMarg, prop::LogNormalMomentMatch, params_old::NamedTuple, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)
    if params_old.α <= 0
        return -Inf
    end
    y = observations(data)
    α_est = nothing
    if length(S_i) >= prop.min_size
        α_est = fit_gamma_shape_moments(view(y, S_i))
    end
    if isnothing(α_est)
        α_est = priors.α_a / priors.α_b
    end
    α_est = max(α_est, 0.01)
    log_α_est = log(α_est)
    log_α = log(params_old.α)
    return logpdf(Normal(log_α_est, prop.σ[1]), log_α) - log_α
end

# --- FixedDistributionProposal ---
function sample_birth_params(::GammaClusterShapeMarg, prop::FixedDistributionProposal, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)
    Q = prop.dists[1]
    α_new = rand(Q)
    return (α = α_new,), logpdf(Q, α_new)
end

function birth_params_logpdf(::GammaClusterShapeMarg, prop::FixedDistributionProposal, params_old::NamedTuple, S_i::Vector{Int}, state::GammaClusterShapeMargState, data::ContinuousData, priors::GammaClusterShapeMargPriors)
    return logpdf(prop.dists[1], params_old.α)
end

# ============================================================================
# Per-parameter dispatch — required by Resample fixed-dim proposal
# ============================================================================
# Resample calls sample_birth_param(model, Val{:α}, inner_proposal, ...)
# rather than sample_birth_params(...) -> NamedTuple.

# --- PriorProposal ---
function sample_birth_param(::GammaClusterShapeMarg, ::Val{:α}, ::PriorProposal,
                            S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    Q = Gamma(priors.α_a, 1/priors.α_b)
    α_new = rand(Q)
    return α_new, logpdf(Q, α_new)
end

function birth_param_logpdf(::GammaClusterShapeMarg, ::Val{:α}, ::PriorProposal,
                            α_val, S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    return logpdf(Gamma(priors.α_a, 1/priors.α_b), α_val)
end

# --- NormalMomentMatch ---
function sample_birth_param(::GammaClusterShapeMarg, ::Val{:α}, prop::NormalMomentMatch,
                            S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    y = observations(data)
    α_est = fit_gamma_shape_moments(view(y, S_i))
    if isnothing(α_est)
        α_est = priors.α_a / priors.α_b
    end
    Q = truncated(Normal(α_est, prop.σ[1]), 0.0, Inf)
    α_new = rand(Q)
    return α_new, logpdf(Q, α_new)
end

function birth_param_logpdf(::GammaClusterShapeMarg, ::Val{:α}, prop::NormalMomentMatch,
                            α_val, S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    y = observations(data)
    α_est = fit_gamma_shape_moments(view(y, S_i))
    if isnothing(α_est)
        α_est = priors.α_a / priors.α_b
    end
    Q = truncated(Normal(α_est, prop.σ[1]), 0.0, Inf)
    return logpdf(Q, α_val)
end

# --- InverseGammaMomentMatch (falls back to prior for the shape parameter) ---
function sample_birth_param(model::GammaClusterShapeMarg, ::Val{:α}, ::InverseGammaMomentMatch,
                            S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    return sample_birth_param(model, Val(:α), PriorProposal(), S_i, state, data, priors)
end

function birth_param_logpdf(model::GammaClusterShapeMarg, ::Val{:α}, ::InverseGammaMomentMatch,
                            α_val, S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    return birth_param_logpdf(model, Val(:α), PriorProposal(), α_val, S_i, state, data, priors)
end

# --- LogNormalMomentMatch ---
function sample_birth_param(::GammaClusterShapeMarg, ::Val{:α}, prop::LogNormalMomentMatch,
                            S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    y = observations(data)
    α_est = nothing
    if length(S_i) >= prop.min_size
        α_est = fit_gamma_shape_moments(view(y, S_i))
    end
    if isnothing(α_est)
        α_est = priors.α_a / priors.α_b
    end
    α_est = max(α_est, 0.01)
    log_α_new = rand(Normal(log(α_est), prop.σ[1]))
    α_new = exp(log_α_new)
    log_q = logpdf(Normal(log(α_est), prop.σ[1]), log_α_new) - log_α_new
    return α_new, log_q
end

function birth_param_logpdf(::GammaClusterShapeMarg, ::Val{:α}, prop::LogNormalMomentMatch,
                            α_val, S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    α_val <= 0 && return -Inf
    y = observations(data)
    α_est = nothing
    if length(S_i) >= prop.min_size
        α_est = fit_gamma_shape_moments(view(y, S_i))
    end
    if isnothing(α_est)
        α_est = priors.α_a / priors.α_b
    end
    α_est = max(α_est, 0.01)
    return logpdf(Normal(log(α_est), prop.σ[1]), log(α_val)) - log(α_val)
end

# --- FixedDistributionProposal ---
function sample_birth_param(::GammaClusterShapeMarg, ::Val{:α}, prop::FixedDistributionProposal,
                            S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    Q = prop.dists[1]
    α_new = rand(Q)
    return α_new, logpdf(Q, α_new)
end

function birth_param_logpdf(::GammaClusterShapeMarg, ::Val{:α}, prop::FixedDistributionProposal,
                            α_val, S_i::Vector{Int}, state::GammaClusterShapeMargState,
                            data::ContinuousData, priors::GammaClusterShapeMargPriors)
    return logpdf(prop.dists[1], α_val)
end

# ============================================================================
# Table Contribution (Marginal Likelihood)
# ============================================================================

"""
    table_contribution(model::GammaClusterShapeMarg, table, state, data, priors)

Compute log-contribution of a table with marginalised cluster rate.
Uses Gamma-Gamma conjugacy to integrate out β.

The marginal likelihood for n observations y_1,...,y_n with shape α is:
    log p(y | α, β_a, β_b) =
        (α - 1) * Σ log(y_i)
        - n * loggamma(α)
        + loggamma(n*α + β_a)
        - loggamma(β_a)
        + β_a * log(β_b)
        - (n*α + β_a) * log(Σy_i + β_b)
"""
function table_contribution(
    ::GammaClusterShapeMarg,
    table::AbstractVector{Int},
    state::GammaClusterShapeMargState,
    data::ContinuousData,
    priors::GammaClusterShapeMargPriors
)
    y = observations(data)
    α = state.α_dict[table]

    if α <= 0
        return -Inf
    end

    n = length(table)
    sum_y = sum(view(y, table))
    sum_log_y = sum(log, view(y, table))

    # Marginal likelihood (integrating out β)
    log_lik = (α - 1) * sum_log_y -
              n * loggamma(α) +
              loggamma(n * α + priors.β_a) -
              loggamma(priors.β_a) +
              priors.β_a * log(priors.β_b) -
              (n * α + priors.β_a) * log(sum_y + priors.β_b)

    # Prior on α_k: α ~ Gamma(α_a, α_b) where α_b is rate parameter
    log_prior_α = logpdf(Gamma(priors.α_a, 1/priors.α_b), α)

    return log_lik + log_prior_α
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::GammaClusterShapeMarg, data, state, priors, log_DDCRP)

Compute full log-posterior for Gamma model with marginalised rate.
"""
function posterior(
    model::GammaClusterShapeMarg,
    data::ContinuousData,
    state::GammaClusterShapeMargState,
    priors::GammaClusterShapeMargPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, table, state, data, priors)
               for table in keys(state.α_dict)) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates - Shape (Metropolis-Hastings on log-scale)
# ============================================================================

"""
    update_α!(model::GammaClusterShapeMarg, state, data, priors; prop_sd=0.5)

Update all cluster shape parameters using Metropolis-Hastings on log-scale.
"""
function update_α!(
    model::GammaClusterShapeMarg,
    state::GammaClusterShapeMargState,
    data::ContinuousData,
    priors::GammaClusterShapeMargPriors;
    prop_sd::Float64 = 0.5
)
    for table in keys(state.α_dict)
        update_α_table!(model, table, state, data, priors; prop_sd=prop_sd)
    end
end

"""
    update_α_table!(model::GammaClusterShapeMarg, table, state, data, priors; prop_sd=0.5)

Update shape parameter for a single table using MH on log-scale.
"""
function update_α_table!(
    model::GammaClusterShapeMarg,
    table::Vector{Int},
    state::GammaClusterShapeMargState,
    data::ContinuousData,
    priors::GammaClusterShapeMargPriors;
    prop_sd::Float64 = 0.5
)
    α_old = state.α_dict[table]

    # Propose on log-scale for positivity
    log_α_old = log(α_old)
    log_α_new = rand(Normal(log_α_old, prop_sd))
    α_new = exp(log_α_new)

    # Compute old contribution
    logpost_old = table_contribution(model, table, state, data, priors)

    # Modify in-place
    state.α_dict[table] = α_new

    # Compute new contribution (reads from modified state)
    logpost_new = table_contribution(model, table, state, data, priors)

    # Jacobian for log-scale proposal: J = α_new / α_old
    # log|J| = log(α_new) - log(α_old) = log_α_new - log_α_old
    log_jacobian = log_α_new - log_α_old

    log_accept_ratio = logpost_new - logpost_old + log_jacobian

    if log(rand()) >= log_accept_ratio
        # Reject: restore old value
        state.α_dict[table] = α_old
    end
end

# ============================================================================
# Update Params Orchestration
# ============================================================================

"""
    update_params!(model::GammaClusterShapeMarg, state, data, priors, tables, log_DDCRP, opts)

Update model parameters (α). Assignment updates are handled
separately by `update_c!` in the main MCMC loop.
"""
function update_params!(
    model::GammaClusterShapeMarg,
    state::GammaClusterShapeMargState,
    data::ContinuousData,
    priors::GammaClusterShapeMargPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    if should_infer(opts, :α)
        update_α!(model, state, data, priors; prop_sd=get_prop_sd(opts, :α))
    end
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::GammaClusterShapeMarg, data, ddcrp_params, priors)

Create initial MCMC state for the model.
Initialises shape parameters using method of moments.
"""
function initialise_state(
    ::GammaClusterShapeMarg,
    data::ContinuousData,
    ddcrp_params::DDCRPParams,
    priors::GammaClusterShapeMargPriors
)
    y = observations(data)
    D = distance_matrix(data)
    n = length(y)

    # Validate data: y > 0 for Gamma
    @assert all(y .> 0) "Gamma model requires strictly positive observations (y > 0)"

    # Initialize assignments from DDCRP prior
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)

    # Initialize α parameters via method of moments
    # For Gamma(α, β): mean = α/β, var = α/β²
    # So: CV = σ/μ = 1/√α → α = 1/CV²
    α_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        key = sort(table)
        data_table = view(y, table)

        μ = mean(data_table)
        σ² = var(data_table; corrected=false)

        if μ > 0 && σ² > 0
            # Method of moments: α = μ²/σ²
            α_est = μ^2 / σ²
            α_dict[key] = max(α_est, 0.1)  # Floor to avoid very small values
        else
            # Fallback: sample from prior
            α_dict[key] = rand(Gamma(priors.α_a, 1/priors.α_b))
        end
    end

    return GammaClusterShapeMargState(c, α_dict)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::GammaClusterShapeMarg, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::GammaClusterShapeMarg, n_samples::Int, n::Int)
    GammaClusterShapeMargSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # α (per observation)
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::GammaClusterShapeMarg, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::GammaClusterShapeMarg,
    state::GammaClusterShapeMargState,
    samples::GammaClusterShapeMargSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    # Store α per observation (each obs gets its cluster's α)
    for (table, α_val) in state.α_dict
        for i in table
            samples.α[iter, i] = α_val
        end
    end
end
