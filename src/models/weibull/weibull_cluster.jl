# ============================================================================
# WeibullCluster - Weibull model with cluster-specific shape and rate parameters
# ============================================================================
#
# Model:
#   y_i | k_c, λ_c ~ Weibull(k_c, 1/λ_c)    for observation i in cluster c
#   k_c ~ Gamma(k_a, 1/k_b)                  (explicit, sampled via MH)
#   λ_c ~ Gamma(λ_a, 1/λ_b)                  (explicit, sampled via MH)
#
# There is no closed-form conjugate prior for the Weibull distribution that
# allows marginalising out either parameter. Both k and λ are sampled
# explicitly using Metropolis-Hastings on log-scale.
#
# Log-likelihood for n observations y_1,...,y_n in cluster c:
#   log p(y | k, λ) =
#       n⋅log(k) + n⋅k⋅log(λ) + (k−1)⋅Σlog(y_i) − λ^k⋅Σy_i^k
#
# Parameters: c (assignments), k_c (cluster shape), λ_c (cluster rate)
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    WeibullCluster <: WeibullModel

Weibull model with cluster-specific shape (`k_c`) and rate (`λ_c`) parameters.
Both parameters are sampled explicitly via Metropolis-Hastings on log-scale.

Parameters:
- k_c: Cluster shape parameters (positive, cluster-level, explicit)
- λ_c: Cluster rate parameters (positive, cluster-level, explicit)
- c: Customer assignments
"""
struct WeibullCluster <: WeibullModel end

# ============================================================================
# State Type
# ============================================================================

"""
    WeibullClusterState{T<:Real} <: AbstractMCMCState{T}

State for WeibullCluster model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `k_dict::Dict{Vector{Int}, T}`: Table -> cluster shape parameter mapping
- `λ_dict::Dict{Vector{Int}, T}`: Table -> cluster rate parameter mapping
"""
mutable struct WeibullClusterState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    k_dict::Dict{Vector{Int}, T}
    λ_dict::Dict{Vector{Int}, T}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    WeibullClusterPriors{T<:Real} <: AbstractPriors

Prior specification for WeibullCluster model.

# Fields
- `k_a::T`: Gamma shape parameter for k prior
- `k_b::T`: Gamma rate parameter for k prior
- `λ_a::T`: Gamma shape parameter for λ prior
- `λ_b::T`: Gamma rate parameter for λ prior
"""
struct WeibullClusterPriors{T<:Real} <: AbstractPriors
    k_a::T
    k_b::T
    λ_a::T
    λ_b::T
end

function WeibullClusterPriors(;
    k_a::Real = 2.0,
    k_b::Real = 1.0,
    λ_a::Real = 2.0,
    λ_b::Real = 1.0
)
    T = promote_type(typeof(k_a), typeof(k_b), typeof(λ_a), typeof(λ_b))
    WeibullClusterPriors{T}(T(k_a), T(k_b), T(λ_a), T(λ_b))
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    WeibullClusterSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for WeibullCluster model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples × n_obs)
- `k::Matrix{T}`: Shape per observation (n_samples × n_obs) - stores cluster k
- `λ::Matrix{T}`: Rate per observation (n_samples × n_obs) - stores cluster λ
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct WeibullClusterSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    k::Matrix{T}
    λ::Matrix{T}
    logpost::Vector{T}
end

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::WeibullClusterState) = (k = state.k_dict, λ = state.λ_dict)

# --- PriorProposal ---
function sample_birth_params(::WeibullCluster, ::PriorProposal, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    Q_k = Gamma(priors.k_a, 1/priors.k_b)
    Q_λ = Gamma(priors.λ_a, 1/priors.λ_b)
    k_new = rand(Q_k)
    λ_new = rand(Q_λ)
    log_q = logpdf(Q_k, k_new) + logpdf(Q_λ, λ_new)
    return (k = k_new, λ = λ_new), log_q
end

function birth_params_logpdf(::WeibullCluster, ::PriorProposal, params_old::NamedTuple, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    return logpdf(Gamma(priors.k_a, 1/priors.k_b), params_old.k) +
           logpdf(Gamma(priors.λ_a, 1/priors.λ_b), params_old.λ)
end

# --- NormalMomentMatch ---
function sample_birth_params(::WeibullCluster, prop::NormalMomentMatch, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    y = observations(data)
    y_S = view(y, S_i)

    k_est = fit_weibull_shape_moments(y_S)
    if isnothing(k_est)
        k_est = priors.k_a / priors.k_b
    end

    λ_est = fit_weibull_rate_moments(y_S, k_est)
    if isnothing(λ_est)
        λ_est = priors.λ_a / priors.λ_b
    end

    σ_k = length(prop.σ) >= 1 ? prop.σ[1] : 0.5
    σ_λ = length(prop.σ) >= 2 ? prop.σ[2] : 0.5

    Q_k = truncated(Normal(k_est, σ_k), 0.0, Inf)
    Q_λ = truncated(Normal(λ_est, σ_λ), 0.0, Inf)

    k_new = rand(Q_k)
    λ_new = rand(Q_λ)
    log_q = logpdf(Q_k, k_new) + logpdf(Q_λ, λ_new)
    return (k = k_new, λ = λ_new), log_q
end

function birth_params_logpdf(::WeibullCluster, prop::NormalMomentMatch, params_old::NamedTuple, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    y = observations(data)
    y_S = view(y, S_i)

    k_est = fit_weibull_shape_moments(y_S)
    if isnothing(k_est)
        k_est = priors.k_a / priors.k_b
    end

    λ_est = fit_weibull_rate_moments(y_S, k_est)
    if isnothing(λ_est)
        λ_est = priors.λ_a / priors.λ_b
    end

    σ_k = length(prop.σ) >= 1 ? prop.σ[1] : 0.5
    σ_λ = length(prop.σ) >= 2 ? prop.σ[2] : 0.5

    Q_k = truncated(Normal(k_est, σ_k), 0.0, Inf)
    Q_λ = truncated(Normal(λ_est, σ_λ), 0.0, Inf)

    return logpdf(Q_k, params_old.k) + logpdf(Q_λ, params_old.λ)
end

# --- LogNormalMomentMatch ---
function sample_birth_params(::WeibullCluster, prop::LogNormalMomentMatch, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    y = observations(data)
    y_S = view(y, S_i)

    k_est = nothing
    λ_est = nothing
    if length(S_i) >= prop.min_size
        k_est = fit_weibull_shape_moments(y_S)
        if !isnothing(k_est)
            λ_est = fit_weibull_rate_moments(y_S, k_est)
        end
    end

    if isnothing(k_est)
        k_est = priors.k_a / priors.k_b
    end
    if isnothing(λ_est)
        λ_est = priors.λ_a / priors.λ_b
    end

    k_est = max(k_est, 0.01)
    λ_est = max(λ_est, 0.01)

    σ_k = length(prop.σ) >= 1 ? prop.σ[1] : 0.5
    σ_λ = length(prop.σ) >= 2 ? prop.σ[2] : 0.5

    log_k_new = rand(Normal(log(k_est), σ_k))
    log_λ_new = rand(Normal(log(λ_est), σ_λ))
    k_new = exp(log_k_new)
    λ_new = exp(log_λ_new)

    # Log-density on original scale: logpdf(Normal, log_x) - log(x)  (Jacobian)
    log_q = logpdf(Normal(log(k_est), σ_k), log_k_new) - log_k_new +
            logpdf(Normal(log(λ_est), σ_λ), log_λ_new) - log_λ_new
    return (k = k_new, λ = λ_new), log_q
end

function birth_params_logpdf(::WeibullCluster, prop::LogNormalMomentMatch, params_old::NamedTuple, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    (params_old.k <= 0 || params_old.λ <= 0) && return -Inf

    y = observations(data)
    y_S = view(y, S_i)

    k_est = nothing
    λ_est = nothing
    if length(S_i) >= prop.min_size
        k_est = fit_weibull_shape_moments(y_S)
        if !isnothing(k_est)
            λ_est = fit_weibull_rate_moments(y_S, k_est)
        end
    end

    if isnothing(k_est)
        k_est = priors.k_a / priors.k_b
    end
    if isnothing(λ_est)
        λ_est = priors.λ_a / priors.λ_b
    end

    k_est = max(k_est, 0.01)
    λ_est = max(λ_est, 0.01)

    σ_k = length(prop.σ) >= 1 ? prop.σ[1] : 0.5
    σ_λ = length(prop.σ) >= 2 ? prop.σ[2] : 0.5

    log_k = log(params_old.k)
    log_λ = log(params_old.λ)

    return logpdf(Normal(log(k_est), σ_k), log_k) - log_k +
           logpdf(Normal(log(λ_est), σ_λ), log_λ) - log_λ
end

# --- InverseGammaMomentMatch (fallback to prior - not suited for Weibull params) ---
function sample_birth_params(::WeibullCluster, prop::InverseGammaMomentMatch, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    return sample_birth_params(WeibullCluster(), PriorProposal(), S_i, state, data, priors)
end

function birth_params_logpdf(::WeibullCluster, prop::InverseGammaMomentMatch, params_old::NamedTuple, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    return birth_params_logpdf(WeibullCluster(), PriorProposal(), params_old, S_i, state, data, priors)
end

# --- FixedDistributionProposal ---
function sample_birth_params(::WeibullCluster, prop::FixedDistributionProposal, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    Q_k = prop.dists[1]
    Q_λ = length(prop.dists) >= 2 ? prop.dists[2] : Gamma(priors.λ_a, 1/priors.λ_b)
    k_new = rand(Q_k)
    λ_new = rand(Q_λ)
    return (k = k_new, λ = λ_new), logpdf(Q_k, k_new) + logpdf(Q_λ, λ_new)
end

function birth_params_logpdf(::WeibullCluster, prop::FixedDistributionProposal, params_old::NamedTuple, S_i::Vector{Int}, state::WeibullClusterState, data::ContinuousData, priors::WeibullClusterPriors)
    Q_k = prop.dists[1]
    Q_λ = length(prop.dists) >= 2 ? prop.dists[2] : Gamma(priors.λ_a, 1/priors.λ_b)
    return logpdf(Q_k, params_old.k) + logpdf(Q_λ, params_old.λ)
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::WeibullCluster, table, state, data, priors)

Compute log-contribution of a table: log-likelihood + log-priors on k and λ.

Log-likelihood for n observations y_1,...,y_n with shape k and rate λ:
    n⋅log(k) + n⋅k⋅log(λ) + (k−1)⋅Σlog(y_i) − λ^k⋅Σy_i^k
"""
function table_contribution(
    ::WeibullCluster,
    table::AbstractVector{Int},
    state::WeibullClusterState,
    data::ContinuousData,
    priors::WeibullClusterPriors
)
    y = observations(data)
    k = state.k_dict[table]
    λ = state.λ_dict[table]

    (k <= 0 || λ <= 0) && return -Inf

    n = length(table)
    y_table = view(y, table)

    any(x -> x <= 0, y_table) && return -Inf

    sum_log_y = sum(log, y_table)
    sum_yk = sum(x -> x^k, y_table)

    log_lik = n * log(k) + n * k * log(λ) + (k - 1) * sum_log_y - λ^k * sum_yk

    log_prior_k = logpdf(Gamma(priors.k_a, 1/priors.k_b), k)
    log_prior_λ = logpdf(Gamma(priors.λ_a, 1/priors.λ_b), λ)

    return log_lik + log_prior_k + log_prior_λ
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::WeibullCluster, data, state, priors, log_DDCRP)

Compute full log-posterior for the Weibull cluster model.
"""
function posterior(
    model::WeibullCluster,
    data::ContinuousData,
    state::WeibullClusterState,
    priors::WeibullClusterPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, table, state, data, priors)
               for table in keys(state.k_dict)) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates - Shape k (Metropolis-Hastings on log-scale)
# ============================================================================

"""
    update_k!(model::WeibullCluster, state, data, priors; prop_sd=0.5)

Update all cluster shape parameters using Metropolis-Hastings on log-scale.
"""
function update_k!(
    model::WeibullCluster,
    state::WeibullClusterState,
    data::ContinuousData,
    priors::WeibullClusterPriors;
    prop_sd::Float64 = 0.5
)
    for table in keys(state.k_dict)
        update_k_table!(model, table, state, data, priors; prop_sd=prop_sd)
    end
end

"""
    update_k_table!(model::WeibullCluster, table, state, data, priors; prop_sd=0.5)

Update shape parameter for a single table using MH on log-scale.
"""
function update_k_table!(
    model::WeibullCluster,
    table::Vector{Int},
    state::WeibullClusterState,
    data::ContinuousData,
    priors::WeibullClusterPriors;
    prop_sd::Float64 = 0.5
)
    k_old = state.k_dict[table]

    log_k_old = log(k_old)
    log_k_new = rand(Normal(log_k_old, prop_sd))
    k_new = exp(log_k_new)

    logpost_old = table_contribution(model, table, state, data, priors)
    state.k_dict[table] = k_new
    logpost_new = table_contribution(model, table, state, data, priors)

    # Jacobian for log-scale proposal
    log_jacobian = log_k_new - log_k_old

    log_accept_ratio = logpost_new - logpost_old + log_jacobian

    if log(rand()) >= log_accept_ratio
        state.k_dict[table] = k_old
    end
end

# ============================================================================
# Parameter Updates - Rate λ (Metropolis-Hastings on log-scale)
# ============================================================================

"""
    update_λ!(model::WeibullCluster, state, data, priors; prop_sd=0.5)

Update all cluster rate parameters using Metropolis-Hastings on log-scale.
"""
function update_λ!(
    model::WeibullCluster,
    state::WeibullClusterState,
    data::ContinuousData,
    priors::WeibullClusterPriors;
    prop_sd::Float64 = 0.5
)
    for table in keys(state.λ_dict)
        update_λ_table!(model, table, state, data, priors; prop_sd=prop_sd)
    end
end

"""
    update_λ_table!(model::WeibullCluster, table, state, data, priors; prop_sd=0.5)

Update rate parameter for a single table using MH on log-scale.
"""
function update_λ_table!(
    model::WeibullCluster,
    table::Vector{Int},
    state::WeibullClusterState,
    data::ContinuousData,
    priors::WeibullClusterPriors;
    prop_sd::Float64 = 0.5
)
    λ_old = state.λ_dict[table]

    log_λ_old = log(λ_old)
    log_λ_new = rand(Normal(log_λ_old, prop_sd))
    λ_new = exp(log_λ_new)

    logpost_old = table_contribution(model, table, state, data, priors)
    state.λ_dict[table] = λ_new
    logpost_new = table_contribution(model, table, state, data, priors)

    # Jacobian for log-scale proposal
    log_jacobian = log_λ_new - log_λ_old

    log_accept_ratio = logpost_new - logpost_old + log_jacobian

    if log(rand()) >= log_accept_ratio
        state.λ_dict[table] = λ_old
    end
end

# ============================================================================
# Update Params Orchestration
# ============================================================================

"""
    update_params!(model::WeibullCluster, state, data, priors, tables, log_DDCRP, opts)

Update model parameters (k and λ). Assignment updates are handled
separately by `update_c!` in the main MCMC loop.
"""
function update_params!(
    model::WeibullCluster,
    state::WeibullClusterState,
    data::ContinuousData,
    priors::WeibullClusterPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    if should_infer(opts, :k)
        update_k!(model, state, data, priors; prop_sd=get_prop_sd(opts, :k))
    end
    if should_infer(opts, :λ)
        update_λ!(model, state, data, priors; prop_sd=get_prop_sd(opts, :λ))
    end
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::WeibullCluster, data, ddcrp_params, priors)

Create initial MCMC state for the WeibullCluster model.
Initialises k via the log-std moment estimator and λ via mean-based estimator.
Falls back to prior draws if moment estimation fails.
"""
function initialise_state(
    ::WeibullCluster,
    data::ContinuousData,
    ddcrp_params::DDCRPParams,
    priors::WeibullClusterPriors
)
    y = observations(data)
    D = distance_matrix(data)

    @assert all(y .> 0) "WeibullCluster model requires strictly positive observations (y > 0)"

    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)

    k_dict = Dict{Vector{Int}, Float64}()
    λ_dict = Dict{Vector{Int}, Float64}()

    for table in tables
        key = sort(table)
        y_table = view(y, table)

        k_est = fit_weibull_shape_moments(y_table)
        if isnothing(k_est)
            k_est = rand(Gamma(priors.k_a, 1/priors.k_b))
        end
        k_est = max(k_est, 0.1)

        λ_est = fit_weibull_rate_moments(y_table, k_est)
        if isnothing(λ_est)
            λ_est = rand(Gamma(priors.λ_a, 1/priors.λ_b))
        end
        λ_est = max(λ_est, 0.01)

        k_dict[key] = k_est
        λ_dict[key] = λ_est
    end

    return WeibullClusterState(c, k_dict, λ_dict)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::WeibullCluster, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::WeibullCluster, n_samples::Int, n::Int)
    WeibullClusterSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # k (per observation)
        zeros(n_samples, n),        # λ (per observation)
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::WeibullCluster, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::WeibullCluster,
    state::WeibullClusterState,
    samples::WeibullClusterSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    for (table, k_val) in state.k_dict
        for i in table
            samples.k[iter, i] = k_val
        end
    end
    for (table, λ_val) in state.λ_dict
        for i in table
            samples.λ[iter, i] = λ_val
        end
    end
end
