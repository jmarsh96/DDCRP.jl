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
# Trait Functions
# ============================================================================

has_latent_rates(::SkewNormalCluster) = false
has_global_dispersion(::SkewNormalCluster) = false
has_cluster_dispersion(::SkewNormalCluster) = false
has_cluster_means(::SkewNormalCluster) = false
has_cluster_rates(::SkewNormalCluster) = false
is_marginalised(::SkewNormalCluster) = false

# Skew normal specific traits
has_latent_augmentation(::SkewNormalCluster) = true
has_cluster_location(::SkewNormalCluster) = true
has_cluster_scale(::SkewNormalCluster) = true
has_cluster_shape(::SkewNormalCluster) = true

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
# Birth Proposal Functions for RJMCMC
# ============================================================================

"""
    sample_birth_proposal(model::SkewNormalCluster, S_i, y, h, priors, opts)

Sample new cluster parameters (ξ, ω, α) for a birth move in RJMCMC.

Returns (ξ_new, ω_new, α_new, log_q_forward) where log_q_forward is the
log proposal density for the Hastings ratio.

Strategy:
- α: Use empirical skewness-based proposal (Normal centered at estimate)
- ω: Moment-matched LogNormal proposal using sample variance and δ
- ξ: Sample from full conditional given h, ω, α (Gibbs-like)

The moment-matching for ω uses: Var(Y) = ω² × (1 - 2δ²/π)
so ω_est = sqrt(s² / (1 - 2δ²/π)) where s² is the sample variance.
"""
function sample_birth_proposal(
    model::SkewNormalCluster,
    S_i::Vector{Int},
    y::AbstractVector,
    h::Vector{<:Real},
    priors::SkewNormalClusterPriors,
    opts::MCMCOptions
)
    n_Si = length(S_i)
    data_Si = view(y, S_i)

    # Step 1: Estimate and sample α from empirical skewness
    if n_Si >= 3
        γ₁ = estimate_skewness(collect(data_Si))
        α_est = alpha_from_skewness(γ₁)
    else
        α_est = 0.0
    end

    σ_α = get(opts.birth_proposal_params, :σ_α, 2.0)
    Q_α = Normal(α_est, σ_α)
    α_new = rand(Q_α)
    log_q_α = logpdf(Q_α, α_new)

    # Step 2: Moment-matched proposal for ω using sample variance
    # Skew normal variance: Var(Y) = ω² × (1 - 2δ²/π)
    δ_new = delta_from_alpha(α_new)
    variance_factor = 1 - 2 * δ_new^2 / π

    s² = var(data_Si; corrected=false)
    s² = max(s², 1e-6)  # Floor to avoid numerical issues

    # ω²_est = s² / variance_factor, so ω_est = sqrt(s² / variance_factor)
    # But variance_factor can be very small for large |α|, so we clamp it
    variance_factor = max(variance_factor, 0.1)
    ω_est = sqrt(s² / variance_factor)
    ω_est = max(ω_est, 0.01)  # Ensure positive

    # Sample ω from LogNormal centered at log(ω_est)
    σ_ω = get(opts.birth_proposal_params, :σ_ω, 0.5)
    log_ω_est = log(ω_est)
    Q_log_ω = Normal(log_ω_est, σ_ω)
    log_ω_new = rand(Q_log_ω)
    ω_new = exp(log_ω_new)
    # log_q_ω = logpdf of LogNormal = logpdf(Normal) - log(ω) (Jacobian)
    log_q_ω = logpdf(Q_log_ω, log_ω_new) - log_ω_new

    # Step 3: Sample ξ from full conditional given h, ω, α
    δ = delta_from_alpha(α_new)
    τ = 1 - δ^2
    y_adj = [y[i] - ω_new * δ * h[i] for i in S_i]

    var_lik = ω_new^2 * τ
    prec_lik = n_Si / var_lik
    prec_prior = 1 / priors.ξ_σ^2
    prec_post = prec_lik + prec_prior
    μ_post = (sum(y_adj) / var_lik + priors.ξ_μ / priors.ξ_σ^2) / prec_post
    σ_post = sqrt(1 / prec_post)

    Q_ξ = Normal(μ_post, σ_post)
    ξ_new = rand(Q_ξ)
    log_q_ξ = logpdf(Q_ξ, ξ_new)

    log_q_forward = log_q_ξ + log_q_ω + log_q_α

    return ξ_new, ω_new, α_new, log_q_forward
end

"""
    birth_proposal_logpdf(model::SkewNormalCluster, ξ, ω, α, S_i, y, h, priors, opts)

Compute log proposal density for given parameters (used in death move for reverse ratio).
Must match the proposal distributions in `sample_birth_proposal`.
"""
function birth_proposal_logpdf(
    model::SkewNormalCluster,
    ξ::Real, ω::Real, α::Real,
    S_i::Vector{Int},
    y::AbstractVector,
    h::Vector{<:Real},
    priors::SkewNormalClusterPriors,
    opts::MCMCOptions
)
    n_Si = length(S_i)
    data_Si = view(y, S_i)

    # Step 1: α proposal density (Normal centered at empirical skewness estimate)
    if n_Si >= 3
        γ₁ = estimate_skewness(collect(data_Si))
        α_est = alpha_from_skewness(γ₁)
    else
        α_est = 0.0
    end
    σ_α = get(opts.birth_proposal_params, :σ_α, 2.0)
    log_q_α = logpdf(Normal(α_est, σ_α), α)

    # Step 2: ω proposal density (LogNormal, moment-matched)
    # Note: We use the GIVEN α to compute δ for the moment-matched estimate,
    # matching the sequential sampling in sample_birth_proposal
    δ_given = delta_from_alpha(α)
    variance_factor = 1 - 2 * δ_given^2 / π
    variance_factor = max(variance_factor, 0.1)

    s² = var(data_Si; corrected=false)
    s² = max(s², 1e-6)
    ω_est = sqrt(s² / variance_factor)
    ω_est = max(ω_est, 0.01)

    σ_ω = get(opts.birth_proposal_params, :σ_ω, 0.5)
    log_ω_est = log(ω_est)
    log_ω = log(ω)
    # LogNormal density: logpdf(Normal(μ, σ), log(x)) - log(x)
    log_q_ω = logpdf(Normal(log_ω_est, σ_ω), log_ω) - log_ω

    # Step 3: ξ proposal density (full conditional given α, ω)
    δ = delta_from_alpha(α)
    τ = 1 - δ^2
    y_adj = [y[i] - ω * δ * h[i] for i in S_i]

    var_lik = ω^2 * τ
    prec_lik = n_Si / var_lik
    prec_prior = 1 / priors.ξ_σ^2
    prec_post = prec_lik + prec_prior
    μ_post = (sum(y_adj) / var_lik + priors.ξ_μ / priors.ξ_σ^2) / prec_post
    σ_post = sqrt(1 / prec_post)

    log_q_ξ = logpdf(Normal(μ_post, σ_post), ξ)

    return log_q_ξ + log_q_ω + log_q_α
end

# ============================================================================
# RJMCMC for Customer Assignments
# ============================================================================

"""
    update_c_rjmcmc!(model::SkewNormalCluster, i, state, data, priors, log_DDCRP, opts)

Update customer assignment using RJMCMC (birth/death/fixed dimension moves).
Returns (move_type::Symbol, j_star::Int, accepted::Bool).
"""
function update_c_rjmcmc!(
    model::SkewNormalCluster,
    i::Int,
    state::SkewNormalClusterState,
    data::ContinuousData,
    priors::SkewNormalClusterPriors,
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    n = length(state.c)
    j_old = state.c[i]
    y = observations(data)

    S_i = get_moving_set(i, state.c)
    table_Si = find_table_for_customer(i, state.ξ_dict)
    ξ_old = state.ξ_dict[table_Si]
    ω_old = state.ω_dict[table_Si]
    α_old = state.α_dict[table_Si]
    table_l = setdiff(table_Si, S_i)

    # Propose new link
    j_star = rand(1:n)

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    # Copy state dictionaries for candidate
    ξ_can = copy(state.ξ_dict)
    ω_can = copy(state.ω_dict)
    α_can = copy(state.α_dict)
    c_can = copy(state.c)
    c_can[i] = j_star

    if !j_old_in_Si && j_star_in_Si
        # ===== BIRTH MOVE =====
        # S_i splits off from its current table to form a new cluster

        # Sample new parameters for the new cluster (S_i)
        ξ_new, ω_new, α_new, log_q_forward = sample_birth_proposal(
            model, S_i, y, state.h, priors, opts
        )

        # Update dictionaries: S_i gets new params, remaining (table_l) keeps old
        ξ_can[sort(S_i)] = ξ_new
        ω_can[sort(S_i)] = ω_new
        α_can[sort(S_i)] = α_new

        if !isempty(table_l)
            ξ_can[sort(table_l)] = state.ξ_dict[table_Si]
            ω_can[sort(table_l)] = state.ω_dict[table_Si]
            α_can[sort(table_l)] = state.α_dict[table_Si]
        end

        delete!(ξ_can, table_Si)
        delete!(ω_can, table_Si)
        delete!(α_can, table_Si)

        lpr = -log_q_forward

        state_can = SkewNormalClusterState(c_can, state.h, ξ_can, ω_can, α_can)
        logpost_current = posterior(model, data, state, priors, log_DDCRP)
        logpost_candidate = posterior(model, data, state_can, priors, log_DDCRP)
        log_α = logpost_candidate - logpost_current + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.ξ_dict); merge!(state.ξ_dict, ξ_can)
            empty!(state.ω_dict); merge!(state.ω_dict, ω_can)
            empty!(state.α_dict); merge!(state.α_dict, α_can)
            return (:birth, j_star, true)
        end
        return (:birth, j_star, false)

    elseif j_old_in_Si && !j_star_in_Si
        # ===== DEATH MOVE =====
        # S_i merges into target table

        table_target = find_table_for_customer(j_star, state.ξ_dict)
        ξ_target = state.ξ_dict[table_target]
        ω_target = state.ω_dict[table_target]
        α_target = state.α_dict[table_target]

        # Merged table gets target's parameters
        merged_table = sort(vcat(table_Si, table_target))
        ξ_can[merged_table] = ξ_target
        ω_can[merged_table] = ω_target
        α_can[merged_table] = α_target

        # Compute reverse proposal density (what would be the birth proposal for old params)
        log_q_reverse = birth_proposal_logpdf(
            model, ξ_old, ω_old, α_old, S_i, y, state.h, priors, opts
        )

        delete!(ξ_can, table_Si)
        delete!(ξ_can, table_target)
        delete!(ω_can, table_Si)
        delete!(ω_can, table_target)
        delete!(α_can, table_Si)
        delete!(α_can, table_target)

        lpr = log_q_reverse

        state_can = SkewNormalClusterState(c_can, state.h, ξ_can, ω_can, α_can)
        logpost_current = posterior(model, data, state, priors, log_DDCRP)
        logpost_candidate = posterior(model, data, state_can, priors, log_DDCRP)
        log_α = logpost_candidate - logpost_current + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.ξ_dict); merge!(state.ξ_dict, ξ_can)
            empty!(state.ω_dict); merge!(state.ω_dict, ω_can)
            empty!(state.α_dict); merge!(state.α_dict, α_can)
            return (:death, j_star, true)
        end
        return (:death, j_star, false)

    else
        # ===== FIXED DIMENSION MOVE =====
        table_old_target = find_table_for_customer(j_old, state.ξ_dict)
        table_new_target = find_table_for_customer(j_star, state.ξ_dict)

        if table_old_target == table_new_target
            # Same table - just relink within table
            state_can = SkewNormalClusterState(c_can, state.h, state.ξ_dict, state.ω_dict, state.α_dict)

            logpost_current = posterior(model, data, state, priors, log_DDCRP)
            logpost_candidate = posterior(model, data, state_can, priors, log_DDCRP)
            log_α = logpost_candidate - logpost_current

            if log(rand()) < log_α
                state.c[i] = j_star
                return (:fixed, j_star, true)
            end
            return (:fixed, j_star, false)
        end

        # Different tables - transfer S_i from old to new table
        new_table_depleted = setdiff(table_old_target, S_i)
        new_table_augmented = sort(vcat(table_new_target, S_i))

        # Keep parameters unchanged (simplest fixed-dim approach)
        ξ_depleted = state.ξ_dict[table_old_target]
        ξ_augmented = state.ξ_dict[table_new_target]
        ω_depleted = state.ω_dict[table_old_target]
        ω_augmented = state.ω_dict[table_new_target]
        α_depleted = state.α_dict[table_old_target]
        α_augmented = state.α_dict[table_new_target]

        if !isempty(new_table_depleted)
            ξ_can[new_table_depleted] = ξ_depleted
            ω_can[new_table_depleted] = ω_depleted
            α_can[new_table_depleted] = α_depleted
        end
        ξ_can[new_table_augmented] = ξ_augmented
        ω_can[new_table_augmented] = ω_augmented
        α_can[new_table_augmented] = α_augmented

        delete!(ξ_can, table_old_target)
        delete!(ξ_can, table_new_target)
        delete!(ω_can, table_old_target)
        delete!(ω_can, table_new_target)
        delete!(α_can, table_old_target)
        delete!(α_can, table_new_target)

        state_can = SkewNormalClusterState(c_can, state.h, ξ_can, ω_can, α_can)
        logpost_current = posterior(model, data, state, priors, log_DDCRP)
        logpost_candidate = posterior(model, data, state_can, priors, log_DDCRP)
        log_α = logpost_candidate - logpost_current

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.ξ_dict); merge!(state.ξ_dict, ξ_can)
            empty!(state.ω_dict); merge!(state.ω_dict, ω_can)
            empty!(state.α_dict); merge!(state.α_dict, α_can)
            return (:fixed, j_star, true)
        end
        return (:fixed, j_star, false)
    end
end

# ============================================================================
# Update Params Orchestration
# ============================================================================

"""
    update_params!(model::SkewNormalCluster, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters in sequence.
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
    diagnostics = Vector{Tuple{Symbol, Int, Int, Bool}}()

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

    # 5. Update assignments c (RJMCMC)
    if should_infer(opts, :c)
        for i in 1:nobs(data)
            move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, log_DDCRP, opts)
            push!(diagnostics, (move_type, i, j_star, accepted))
        end
    end

    return diagnostics
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

# ============================================================================
# Helper function to find table for customer (used in RJMCMC)
# ============================================================================

"""
    find_table_for_customer(i::Int, param_dict::Dict{Vector{Int}, T}) where T

Find the table (key) that contains customer i.
"""
function find_table_for_customer(i::Int, param_dict::Dict{Vector{Int}, T}) where T
    for table in keys(param_dict)
        if i in table
            return table
        end
    end
    error("Customer $i not found in any table")
end
