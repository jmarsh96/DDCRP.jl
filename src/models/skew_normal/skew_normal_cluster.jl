# ============================================================================
# SkewNormalCluster - Skew Normal with cluster-specific location, scale, shape
# ============================================================================
#
# Model:
#   y_i | ξ_k, ω_k, α_k ~ SkewNormal(ξ_k, ω_k, α_k)  for observation i in cluster k
#   ξ_k ~ Normal(ξ_μ, ξ_σ)                           (cluster location)
#   ω_k² ~ InverseGamma(ω_a, ω_b)                    (cluster scale squared)
#   α_k ~ Normal(α_μ, α_σ)                           (cluster shape/skewness)
#
# The skew normal PDF is:
#   f(y | ξ, ω, α) = (2/ω) × φ((y-ξ)/ω) × Φ(α(y-ξ)/ω)
#
# Data augmentation scheme:
#   y_i = ξ_k + ω_k × (δ_k × h_i + √(1-δ_k²) × ε_i)
#   where h_i = |z_i| with z_i ~ N(0,1), and δ_k = α_k / √(1 + α_k²)
#
# Parameters: c (assignments), h (latent), ξ_k (location), ω_k (scale), α_k (shape)
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    SkewNormalCluster <: SkewNormalModel

Skew Normal model with cluster-specific location (ξ), scale (ω), and shape (α).

Uses data augmentation for efficient Gibbs sampling of ξ and conjugate updates.
Shape parameter α is updated via Metropolis-Hastings.
"""
struct SkewNormalCluster <: SkewNormalModel end

# ============================================================================
# State Type
# ============================================================================

"""
    SkewNormalClusterState{T<:Real} <: AbstractMCMCState{T}

State for SkewNormalCluster model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `h::Vector{T}`: Latent augmentation variables (one per observation)
- `ξ_dict::Dict{Vector{Int}, T}`: Table -> location parameter mapping
- `ω_dict::Dict{Vector{Int}, T}`: Table -> scale parameter mapping
- `α_dict::Dict{Vector{Int}, T}`: Table -> shape parameter mapping
"""
mutable struct SkewNormalClusterState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    h::Vector{T}
    ξ_dict::Dict{Vector{Int}, T}
    ω_dict::Dict{Vector{Int}, T}
    α_dict::Dict{Vector{Int}, T}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    SkewNormalClusterPriors{T<:Real} <: AbstractPriors

Prior specification for SkewNormalCluster model.

# Fields
- `ξ_μ::T`: Normal prior mean for location ξ
- `ξ_σ::T`: Normal prior std for location ξ
- `ω_a::T`: InverseGamma shape parameter for ω² (scale squared)
- `ω_b::T`: InverseGamma scale parameter for ω²
- `α_μ::T`: Normal prior mean for shape α (default: 0 = symmetric)
- `α_σ::T`: Normal prior std for shape α
"""
struct SkewNormalClusterPriors{T<:Real} <: AbstractPriors
    ξ_μ::T
    ξ_σ::T
    ω_a::T
    ω_b::T
    α_μ::T
    α_σ::T
end

# Default constructor with reasonable defaults
function SkewNormalClusterPriors(;
    ξ_μ::Real = 0.0,
    ξ_σ::Real = 10.0,
    ω_a::Real = 2.0,
    ω_b::Real = 1.0,
    α_μ::Real = 0.0,
    α_σ::Real = 5.0
)
    T = promote_type(typeof(ξ_μ), typeof(ξ_σ), typeof(ω_a), typeof(ω_b), typeof(α_μ), typeof(α_σ))
    SkewNormalClusterPriors{T}(T(ξ_μ), T(ξ_σ), T(ω_a), T(ω_b), T(α_μ), T(α_σ))
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    SkewNormalClusterSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for SkewNormalCluster model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `h::Matrix{T}`: Latent augmentation variables (n_samples x n_obs)
- `ξ::Matrix{T}`: Location per observation (n_samples x n_obs)
- `ω::Matrix{T}`: Scale per observation (n_samples x n_obs)
- `α::Matrix{T}`: Shape per observation (n_samples x n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct SkewNormalClusterSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    h::Matrix{T}
    ξ::Matrix{T}
    ω::Matrix{T}
    α::Matrix{T}
    logpost::Vector{T}
end


# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::SkewNormalClusterState) = (ξ = state.ξ_dict, ω = state.ω_dict, α = state.α_dict)
copy_cluster_param_dicts(state::SkewNormalClusterState) = (ξ = copy(state.ξ_dict), ω = copy(state.ω_dict), α = copy(state.α_dict))

function make_candidate_state(::SkewNormalCluster, state::SkewNormalClusterState,
                              c_can::Vector{Int}, params_can::NamedTuple)
    SkewNormalClusterState(c_can, state.h, params_can.ξ, params_can.ω, params_can.α)
end

function commit_params!(state::SkewNormalClusterState, params_can::NamedTuple)
    empty!(state.ξ_dict); merge!(state.ξ_dict, params_can.ξ)
    empty!(state.ω_dict); merge!(state.ω_dict, params_can.ω)
    empty!(state.α_dict); merge!(state.α_dict, params_can.α)
end

# --- PriorProposal ---
function sample_birth_params(::SkewNormalCluster, ::PriorProposal,
                             S_i::Vector{Int}, state::SkewNormalClusterState,
                             data::ContinuousData, priors::SkewNormalClusterPriors)
    Q_ξ = Normal(priors.ξ_μ, priors.ξ_σ)
    Q_ω² = InverseGamma(priors.ω_a, priors.ω_b)
    Q_α = Normal(priors.α_μ, priors.α_σ)
    ξ_new = rand(Q_ξ)
    ω²_new = rand(Q_ω²)
    ω_new = sqrt(ω²_new)
    α_new = rand(Q_α)
    # log_q includes Jacobian for ω² → ω transform: p(ω) = p(ω²) × 2ω
    log_q = logpdf(Q_ξ, ξ_new) + logpdf(Q_ω², ω²_new) + log(2 * ω_new) + logpdf(Q_α, α_new)
    return (ξ = ξ_new, ω = ω_new, α = α_new), log_q
end

function birth_params_logpdf(::SkewNormalCluster, ::PriorProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::SkewNormalClusterState, data::ContinuousData,
                             priors::SkewNormalClusterPriors)
    Q_ξ = Normal(priors.ξ_μ, priors.ξ_σ)
    Q_ω² = InverseGamma(priors.ω_a, priors.ω_b)
    Q_α = Normal(priors.α_μ, priors.α_σ)
    return logpdf(Q_ξ, params_old.ξ) + logpdf(Q_ω², params_old.ω^2) + log(2 * params_old.ω) + logpdf(Q_α, params_old.α)
end

# --- NormalMomentMatch ---
function sample_birth_params(::SkewNormalCluster, prop::NormalMomentMatch,
                             S_i::Vector{Int}, state::SkewNormalClusterState,
                             data::ContinuousData, priors::SkewNormalClusterPriors)
    y = observations(data)
    data_Si = view(y, S_i)
    # ξ: Normal centered at empirical mean
    Q_ξ = Normal(mean(data_Si), prop.σ[1])
    ξ_new = rand(Q_ξ)
    # ω: LogNormal centered at empirical std
    log_ω_est = log(max(std(data_Si; corrected=false), 0.01))
    Q_log_ω = Normal(log_ω_est, prop.σ[2])
    log_ω_new = rand(Q_log_ω)
    ω_new = exp(log_ω_new)
    # α: Normal centered at skewness estimate
    α_est = length(S_i) >= 3 ? alpha_from_skewness(estimate_skewness(collect(data_Si))) : 0.0
    Q_α = Normal(α_est, prop.σ[3])
    α_new = rand(Q_α)
    log_q = logpdf(Q_ξ, ξ_new) + logpdf(Q_log_ω, log_ω_new) - log_ω_new + logpdf(Q_α, α_new)
    return (ξ = ξ_new, ω = ω_new, α = α_new), log_q
end

function birth_params_logpdf(::SkewNormalCluster, prop::NormalMomentMatch,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::SkewNormalClusterState, data::ContinuousData,
                             priors::SkewNormalClusterPriors)
    y = observations(data)
    data_Si = view(y, S_i)
    Q_ξ = Normal(mean(data_Si), prop.σ[1])
    log_ω_est = log(max(std(data_Si; corrected=false), 0.01))
    Q_log_ω = Normal(log_ω_est, prop.σ[2])
    α_est = length(S_i) >= 3 ? alpha_from_skewness(estimate_skewness(collect(data_Si))) : 0.0
    Q_α = Normal(α_est, prop.σ[3])
    log_ω = log(params_old.ω)
    return logpdf(Q_ξ, params_old.ξ) + logpdf(Q_log_ω, log_ω) - log_ω + logpdf(Q_α, params_old.α)
end

# --- LogNormalMomentMatch ---
function sample_birth_params(::SkewNormalCluster, prop::LogNormalMomentMatch,
                             S_i::Vector{Int}, state::SkewNormalClusterState,
                             data::ContinuousData, priors::SkewNormalClusterPriors)
    y = observations(data)
    data_Si = view(y, S_i)
    # ξ: Normal centered at empirical mean (ξ can be negative, no log transform)
    Q_ξ = Normal(mean(data_Si), prop.σ[1])
    ξ_new = rand(Q_ξ)
    # ω: LogNormal
    log_ω_est = log(max(std(data_Si; corrected=false), 0.01))
    Q_log_ω = Normal(log_ω_est, prop.σ[2])
    log_ω_new = rand(Q_log_ω)
    ω_new = exp(log_ω_new)
    # α: Normal (α can be negative)
    α_est = length(S_i) >= 3 ? alpha_from_skewness(estimate_skewness(collect(data_Si))) : 0.0
    Q_α = Normal(α_est, prop.σ[3])
    α_new = rand(Q_α)
    log_q = logpdf(Q_ξ, ξ_new) + logpdf(Q_log_ω, log_ω_new) - log_ω_new + logpdf(Q_α, α_new)
    return (ξ = ξ_new, ω = ω_new, α = α_new), log_q
end

function birth_params_logpdf(::SkewNormalCluster, prop::LogNormalMomentMatch,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::SkewNormalClusterState, data::ContinuousData,
                             priors::SkewNormalClusterPriors)
    y = observations(data)
    data_Si = view(y, S_i)
    Q_ξ = Normal(mean(data_Si), prop.σ[1])
    log_ω_est = log(max(std(data_Si; corrected=false), 0.01))
    Q_log_ω = Normal(log_ω_est, prop.σ[2])
    α_est = length(S_i) >= 3 ? alpha_from_skewness(estimate_skewness(collect(data_Si))) : 0.0
    Q_α = Normal(α_est, prop.σ[3])
    log_ω = log(params_old.ω)
    return logpdf(Q_ξ, params_old.ξ) + logpdf(Q_log_ω, log_ω) - log_ω + logpdf(Q_α, params_old.α)
end

# --- FixedDistributionProposal ---
function sample_birth_params(::SkewNormalCluster, prop::FixedDistributionProposal,
                             S_i::Vector{Int}, state::SkewNormalClusterState,
                             data::ContinuousData, priors::SkewNormalClusterPriors)
    ξ_new = rand(prop.dists[1])
    ω_new = rand(prop.dists[2])
    α_new = rand(prop.dists[3])
    log_q = logpdf(prop.dists[1], ξ_new) + logpdf(prop.dists[2], ω_new) + logpdf(prop.dists[3], α_new)
    return (ξ = ξ_new, ω = ω_new, α = α_new), log_q
end

function birth_params_logpdf(::SkewNormalCluster, prop::FixedDistributionProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::SkewNormalClusterState, data::ContinuousData,
                             priors::SkewNormalClusterPriors)
    return logpdf(prop.dists[1], params_old.ξ) + logpdf(prop.dists[2], params_old.ω) + logpdf(prop.dists[3], params_old.α)
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::SkewNormalCluster, table, state, data, priors)

Compute log-contribution of a table (cluster) to the posterior.
Includes both log-likelihood and log-priors for cluster parameters.
"""
function table_contribution(
    ::SkewNormalCluster,
    table::AbstractVector{Int},
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors
)
    y = observations(data)
    key = sort(table)
    ξ = state.ξ_dict[key]
    ω = state.ω_dict[key]
    α = state.α_dict[key]

    # Skew normal log-likelihood for all observations in table
    log_lik = sum(skewnormal_logpdf(y[i], ξ, ω, α) for i in table)

    # Priors
    log_prior_ξ = logpdf(Normal(priors.ξ_μ, priors.ξ_σ), ξ)
    # Prior on ω²: ω² ~ InverseGamma(ω_a, ω_b)
    # Need to include Jacobian for transformation: p(ω) = p(ω²) × |d(ω²)/dω| = p(ω²) × 2ω
    log_prior_ω² = logpdf(InverseGamma(priors.ω_a, priors.ω_b), ω^2)
    log_jacobian_ω = log(2 * ω)
    log_prior_α = logpdf(Normal(priors.α_μ, priors.α_σ), α)

    return log_lik + log_prior_ξ + log_prior_ω² + log_jacobian_ω + log_prior_α
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::SkewNormalCluster, data, state, priors, log_DDCRP)

Compute full log-posterior for the skew normal model.
"""
function posterior(
    model::SkewNormalCluster,
    data::ContinuousData,
    state::SkewNormalClusterState,
    priors::SkewNormalClusterPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.ξ_dict)) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates - Latent h (Gibbs)
# ============================================================================

"""
    update_h!(model::SkewNormalCluster, state, data, priors)

Update all latent augmentation variables h using Gibbs sampling.
Each h_i is sampled from its full conditional (truncated normal).
"""
function update_h!(
    model::SkewNormalCluster,
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors
)
    y = observations(data)
    for (table, ξ) in state.ξ_dict
        ω = state.ω_dict[table]
        α = state.α_dict[table]
        for i in table
            state.h[i] = sample_h_conditional(y[i], ξ, ω, α)
        end
    end
end

# ============================================================================
# Parameter Updates - Location ξ (Gibbs)
# ============================================================================

"""
    update_ξ!(model::SkewNormalCluster, state, data, priors)

Update all cluster locations using Gibbs sampling (conjugate Normal update).
"""
function update_ξ!(
    model::SkewNormalCluster,
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors
)
    for table in keys(state.ξ_dict)
        update_ξ_table!(model, table, state, data, priors)
    end
end

"""
    update_ξ_table!(model::SkewNormalCluster, table, state, data, priors)

Update location parameter for a single table using Gibbs sampling.

Given the augmented model:
    y_i | h_i, ξ, ω, α ~ N(ξ + ω × δ × h_i, ω² × τ)

where τ = 1 - δ², the conditional posterior for ξ is:
    ξ | y, h, ω, α ~ N(μ_post, σ_post²)
"""
function update_ξ_table!(
    model::SkewNormalCluster,
    table::Vector{Int},
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors
)
    y = observations(data)
    key = sort(table)
    ω = state.ω_dict[key]
    α = state.α_dict[key]

    δ = delta_from_alpha(α)
    τ = 1 - δ^2

    # Adjusted observations: y_adj_i = y_i - ω × δ × h_i
    # These have mean ξ under the augmented model
    y_adj = [y[i] - ω * δ * state.h[i] for i in table]

    n_k = length(table)
    sum_y_adj = sum(y_adj)

    # Likelihood precision and prior precision
    var_lik = ω^2 * τ
    prec_lik = n_k / var_lik
    prec_prior = 1 / priors.ξ_σ^2

    # Posterior parameters
    prec_post = prec_lik + prec_prior
    μ_post = (sum_y_adj / var_lik + priors.ξ_μ / priors.ξ_σ^2) / prec_post
    σ_post = sqrt(1 / prec_post)

    state.ξ_dict[key] = rand(Normal(μ_post, σ_post))
end

# ============================================================================
# Parameter Updates - Scale ω (Metropolis-Hastings on log-scale)
# ============================================================================

"""
    update_ω!(model::SkewNormalCluster, state, data, priors; prop_sd=0.3)

Update all cluster scales using Metropolis-Hastings on log-scale.
"""
function update_ω!(
    model::SkewNormalCluster,
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors;
    prop_sd::Float64 = 0.3
)
    for table in keys(state.ω_dict)
        update_ω_table!(model, table, state, data, priors; prop_sd=prop_sd)
    end
end

"""
    update_ω_table!(model::SkewNormalCluster, table, state, data, priors; prop_sd=0.3)

Update scale parameter for a single table using MH on log-scale.
"""
function update_ω_table!(
    model::SkewNormalCluster,
    table::Vector{Int},
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors;
    prop_sd::Float64 = 0.3
)
    key = sort(table)
    ω_old = state.ω_dict[key]

    # Propose on log-scale for positivity
    log_ω_old = log(ω_old)
    log_ω_new = rand(Normal(log_ω_old, prop_sd))
    ω_new = exp(log_ω_new)

    # Create candidate state
    ω_dict_can = copy(state.ω_dict)
    ω_dict_can[key] = ω_new
    state_can = SkewNormalClusterState(state.c, state.h, state.ξ_dict, ω_dict_can, state.α_dict)

    # Log-posterior ratio (table contribution only since rest doesn't change)
    logpost_old = table_contribution(model, table, state, data, priors)
    logpost_new = table_contribution(model, table, state_can, data, priors)

    # Jacobian for log-scale proposal: J = ω_new / ω_old (ratio of |dω/d(log ω)|)
    log_jacobian = log_ω_new - log_ω_old

    log_accept_ratio = logpost_new - logpost_old + log_jacobian

    if log(rand()) < log_accept_ratio
        state.ω_dict[key] = ω_new
    end
end

# ============================================================================
# Parameter Updates - Shape α (Metropolis-Hastings)
# ============================================================================

"""
    update_α!(model::SkewNormalCluster, state, data, priors; prop_sd=0.5)

Update all cluster shapes using Metropolis-Hastings.
"""
function update_α!(
    model::SkewNormalCluster,
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors;
    prop_sd::Float64 = 0.5
)
    for table in keys(state.α_dict)
        update_α_table!(model, table, state, data, priors; prop_sd=prop_sd)
    end
end

"""
    update_α_table!(model::SkewNormalCluster, table, state, data, priors; prop_sd=0.5)

Update shape parameter for a single table using Metropolis-Hastings.
"""
function update_α_table!(
    model::SkewNormalCluster,
    table::Vector{Int},
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors;
    prop_sd::Float64 = 0.5
)
    key = sort(table)
    α_old = state.α_dict[key]

    # Symmetric Normal proposal
    α_new = rand(Normal(α_old, prop_sd))

    # Create candidate state
    α_dict_can = copy(state.α_dict)
    α_dict_can[key] = α_new
    state_can = SkewNormalClusterState(state.c, state.h, state.ξ_dict, state.ω_dict, α_dict_can)

    # Log-posterior ratio
    logpost_old = table_contribution(model, table, state, data, priors)
    logpost_new = table_contribution(model, table, state_can, data, priors)

    log_accept_ratio = logpost_new - logpost_old

    if log(rand()) < log_accept_ratio
        state.α_dict[key] = α_new
    end
end

# ============================================================================
# Update Params Orchestration
# ============================================================================

"""
    update_params!(model::SkewNormalCluster, state, data, priors, tables, log_DDCRP, opts)

Update model parameters (h, ξ, ω, α). Assignment updates are handled
separately by `update_c!` in the main MCMC loop.
"""
function update_params!(
    model::SkewNormalCluster,
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    # 1. Update latent h (Gibbs)
    if should_infer(opts, :h)
        update_h!(model, state, data, priors)
    end

    # 2. Update location ξ (Gibbs)
    if should_infer(opts, :ξ)
        update_ξ!(model, state, data, priors)
    end

    # 3. Update scale ω (MH)
    if should_infer(opts, :ω)
        update_ω!(model, state, data, priors; prop_sd=get_prop_sd(opts, :ω))
    end

    # 4. Update shape α (MH)
    if should_infer(opts, :α)
        update_α!(model, state, data, priors; prop_sd=get_prop_sd(opts, :α))
    end
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::SkewNormalCluster, data, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::SkewNormalCluster,
    data::ContinuousData,
    ddcrp_params::DDCRPParams,
    priors::SkewNormalClusterPriors
)
    y = observations(data)
    D = distance_matrix(data)
    n = length(y)

    # Initialize assignments from DDCRP prior
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)

    # Initialize parameter dictionaries
    ξ_dict = Dict{Vector{Int}, Float64}()
    ω_dict = Dict{Vector{Int}, Float64}()
    α_dict = Dict{Vector{Int}, Float64}()

    for table in tables
        key = sort(table)
        data_table = view(y, table)

        # Initialize ξ to cluster mean
        ξ_dict[key] = mean(data_table)

        # Initialize ω to cluster std (with floor)
        ω_dict[key] = max(std(data_table; corrected=false), 0.1)

        # Initialize α based on skewness
        if length(table) >= 3
            γ₁ = estimate_skewness(collect(data_table))
            α_dict[key] = alpha_from_skewness(γ₁)
        else
            α_dict[key] = 0.0
        end
    end

    # Initialize latent h from prior (half-normal)
    h = abs.(randn(n))

    return SkewNormalClusterState(c, h, ξ_dict, ω_dict, α_dict)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::SkewNormalCluster, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::SkewNormalCluster, n_samples::Int, n::Int)
    SkewNormalClusterSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # h
        zeros(n_samples, n),        # ξ (per observation)
        zeros(n_samples, n),        # ω (per observation)
        zeros(n_samples, n),        # α (per observation)
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::SkewNormalCluster, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::SkewNormalCluster,
    state::SkewNormalClusterState,
    samples::SkewNormalClusterSamples,
    iter::Int
)
    n = length(state.c)
    samples.c[iter, :] = state.c
    samples.h[iter, :] = state.h

    # Store cluster parameters per observation
    for (table, ξ_val) in state.ξ_dict
        ω_val = state.ω_dict[table]
        α_val = state.α_dict[table]
        for i in table
            samples.ξ[iter, i] = ξ_val
            samples.ω[iter, i] = ω_val
            samples.α[iter, i] = α_val
        end
    end
end

