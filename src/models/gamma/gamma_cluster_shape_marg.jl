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
# Trait Functions
# ============================================================================

has_latent_rates(::GammaClusterShapeMarg) = false
has_global_dispersion(::GammaClusterShapeMarg) = false
has_cluster_dispersion(::GammaClusterShapeMarg) = false
has_cluster_means(::GammaClusterShapeMarg) = false
has_cluster_rates(::GammaClusterShapeMarg) = false
has_cluster_probs(::GammaClusterShapeMarg) = false
has_cluster_shape(::GammaClusterShapeMarg) = true
is_marginalised(::GammaClusterShapeMarg) = false  # NOT fully marginalised - has explicit α_k

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
    key = sort(table)
    α = state.α_dict[key]

    if α <= 0
        return -Inf
    end

    n = length(table)
    sum_y = sum(view(y, table))
    sum_log_y = sum(log.(view(y, table)))

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
    return sum(table_contribution(model, sort(table), state, data, priors)
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
    key = sort(table)
    α_old = state.α_dict[key]

    # Propose on log-scale for positivity
    log_α_old = log(α_old)
    log_α_new = rand(Normal(log_α_old, prop_sd))
    α_new = exp(log_α_new)

    # Create candidate state
    α_dict_can = copy(state.α_dict)
    α_dict_can[key] = α_new
    state_can = GammaClusterShapeMargState(state.c, α_dict_can)

    # Log-posterior ratio
    logpost_old = table_contribution(model, table, state, data, priors)
    logpost_new = table_contribution(model, table, state_can, data, priors)

    # Jacobian for log-scale proposal: J = α_new / α_old
    # log|J| = log(α_new) - log(α_old) = log_α_new - log_α_old
    log_jacobian = log_α_new - log_α_old

    log_accept_ratio = logpost_new - logpost_old + log_jacobian

    if log(rand()) < log_accept_ratio
        state.α_dict[key] = α_new
    end
end

# ============================================================================
# Birth Proposal Functions
# ============================================================================

"""
    sample_proposal(model::GammaClusterShapeMarg, Q_α::UnivariateDistribution)

Sample a new shape parameter from a UnivariateDistribution.
Returns (α_new, log_q) where log_q is the log proposal density.

Note: If Q_α has support on ℝ (e.g., Normal), we reject negative samples
by returning -Inf for log_q, which guarantees MH rejection.
"""
function sample_proposal(::GammaClusterShapeMarg, Q_α::UnivariateDistribution)
    α_new = rand(Q_α)
    # Gamma shape must be positive - if proposal is invalid, return -Inf log density
    if α_new <= 0
        return α_new, -Inf
    end
    log_q = logpdf(Q_α, α_new)
    return α_new, log_q
end

"""
    proposal_logpdf(model::GammaClusterShapeMarg, Q_α::UnivariateDistribution, α)

Compute the log density of a UnivariateDistribution proposal at α.
Returns -Inf for non-positive α since Gamma shape must be positive.
"""
function proposal_logpdf(::GammaClusterShapeMarg, Q_α::UnivariateDistribution, α)
    if α <= 0
        return -Inf
    end
    return logpdf(Q_α, α)
end

"""
    sample_proposal(model::GammaClusterShapeMarg, prop::MomentMatchedLogNormalProposal, S_i, y, priors)

Sample a new shape parameter using moment-matched LogNormal proposal.
Returns (α_new, log_q) where log_q is the log proposal density.
"""
function sample_proposal(
    ::GammaClusterShapeMarg,
    prop::MomentMatchedLogNormalProposal,
    S_i::Vector{Int},
    y::AbstractVector,
    priors::GammaClusterShapeMargPriors
)
    return sample_proposal(prop, S_i, y, priors)
end

"""
    proposal_logpdf(model::GammaClusterShapeMarg, prop::MomentMatchedLogNormalProposal, α, S_i, y, priors)

Compute the log density of the moment-matched LogNormal proposal at α.
"""
function proposal_logpdf(
    ::GammaClusterShapeMarg,
    prop::MomentMatchedLogNormalProposal,
    α::Real,
    S_i::Vector{Int},
    y::AbstractVector,
    priors::GammaClusterShapeMargPriors
)
    return proposal_logpdf(prop, α, S_i, y, priors)
end

# ============================================================================
# Internal dispatch helpers for birth proposals
# ============================================================================

"""
    _sample_birth_proposal(model, prop::MomentMatchedLogNormalProposal, S_i, y, priors)

Sample using moment-matched LogNormal proposal (data-informed).
"""
function _sample_birth_proposal(
    model::GammaClusterShapeMarg,
    prop::MomentMatchedLogNormalProposal,
    S_i::Vector{Int},
    y::AbstractVector,
    priors::GammaClusterShapeMargPriors
)
    return sample_proposal(prop, S_i, y, priors)
end

"""
    _sample_birth_proposal(model, prop::PriorProposal, S_i, y, priors)

Sample from prior Gamma distribution.
"""
function _sample_birth_proposal(
    model::GammaClusterShapeMarg,
    prop::PriorProposal,
    S_i::Vector{Int},
    y::AbstractVector,
    priors::GammaClusterShapeMargPriors
)
    Q = Gamma(priors.α_a, 1/priors.α_b)
    return sample_proposal(model, Q)
end

"""
    _sample_birth_proposal(model, Q::UnivariateDistribution, S_i, y, priors)

Sample from a generic UnivariateDistribution.
"""
function _sample_birth_proposal(
    model::GammaClusterShapeMarg,
    Q::UnivariateDistribution,
    S_i::Vector{Int},
    y::AbstractVector,
    priors::GammaClusterShapeMargPriors
)
    return sample_proposal(model, Q)
end

"""
    _proposal_logpdf(model, prop::MomentMatchedLogNormalProposal, α, S_i, y, priors)

Evaluate moment-matched LogNormal proposal density.
"""
function _proposal_logpdf(
    model::GammaClusterShapeMarg,
    prop::MomentMatchedLogNormalProposal,
    α::Real,
    S_i::Vector{Int},
    y::AbstractVector,
    priors::GammaClusterShapeMargPriors
)
    return proposal_logpdf(prop, α, S_i, y, priors)
end

"""
    _proposal_logpdf(model, prop::PriorProposal, α, S_i, y, priors)

Evaluate prior Gamma proposal density.
"""
function _proposal_logpdf(
    model::GammaClusterShapeMarg,
    prop::PriorProposal,
    α::Real,
    S_i::Vector{Int},
    y::AbstractVector,
    priors::GammaClusterShapeMargPriors
)
    Q = Gamma(priors.α_a, 1/priors.α_b)
    return proposal_logpdf(model, Q, α)
end

"""
    _proposal_logpdf(model, Q::UnivariateDistribution, α, S_i, y, priors)

Evaluate generic UnivariateDistribution proposal density.
"""
function _proposal_logpdf(
    model::GammaClusterShapeMarg,
    Q::UnivariateDistribution,
    α::Real,
    S_i::Vector{Int},
    y::AbstractVector,
    priors::GammaClusterShapeMargPriors
)
    return proposal_logpdf(model, Q, α)
end

# ============================================================================
# RJMCMC Update
# ============================================================================

"""
    update_c_rjmcmc!(model::GammaClusterShapeMarg, i, state, data, priors, log_DDCRP, opts)

Update customer assignment using RJMCMC (birth/death/fixed dimension moves).

Supports different birth proposals via opts.birth_proposal:
- :prior (default) - Sample from Gamma prior
- :moment_matched_lognormal - LogNormal centered at method-of-moments estimate
- UnivariateDistribution - Sample directly from provided distribution
"""
function update_c_rjmcmc!(
    model::GammaClusterShapeMarg,
    i::Int,
    state::GammaClusterShapeMargState,
    data::ContinuousData,
    priors::GammaClusterShapeMargPriors,
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    n = length(state.c)
    j_old = state.c[i]
    y = observations(data)

    # Build birth proposal from options
    birth_prop = build_birth_proposal(opts)

    S_i = get_moving_set(i, state.c)
    table_Si = find_table_for_customer(i, state.α_dict)
    α_old = state.α_dict[table_Si]
    table_l = setdiff(table_Si, S_i)

    j_star = rand(1:n)

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    α_can = copy(state.α_dict)
    c_can = copy(state.c)
    c_can[i] = j_star

    if !j_old_in_Si && j_star_in_Si
        # ===== BIRTH MOVE =====
        # Moving set S_i stays where it is, customer i moves to form new cluster
        # Sample new α for the moving set S_i
        α_new, log_q_forward = _sample_birth_proposal(model, birth_prop, S_i, y, priors)

        α_can[sort(S_i)] = α_new
        if !isempty(table_l)
            α_can[sort(table_l)] = state.α_dict[table_Si]
        end
        delete!(α_can, table_Si)

        lpr = -log_q_forward  # Hastings ratio for birth

        state_can = GammaClusterShapeMargState(c_can, α_can)
        logpost_current = posterior(model, data, state, priors, log_DDCRP)
        logpost_candidate = posterior(model, data, state_can, priors, log_DDCRP)
        log_α_accept = logpost_candidate - logpost_current + lpr

        if log(rand()) < log_α_accept
            state.c[i] = j_star
            empty!(state.α_dict)
            merge!(state.α_dict, α_can)
            return (:birth, j_star, true)
        end
        return (:birth, j_star, false)

    elseif j_old_in_Si && !j_star_in_Si
        # ===== DEATH MOVE =====
        # Moving set S_i merges with target cluster
        table_target = find_table_for_customer(j_star, state.α_dict)
        α_target = state.α_dict[table_target]

        merged_table = sort(vcat(table_Si, table_target))
        α_can[merged_table] = α_target

        # Reverse proposal density (for Hastings ratio)
        log_q_reverse = _proposal_logpdf(model, birth_prop, α_old, S_i, y, priors)

        delete!(α_can, table_Si)
        delete!(α_can, table_target)

        lpr = log_q_reverse  # Hastings ratio for death

        state_can = GammaClusterShapeMargState(c_can, α_can)
        logpost_current = posterior(model, data, state, priors, log_DDCRP)
        logpost_candidate = posterior(model, data, state_can, priors, log_DDCRP)
        log_α_accept = logpost_candidate - logpost_current + lpr

        if log(rand()) < log_α_accept
            state.c[i] = j_star
            empty!(state.α_dict)
            merge!(state.α_dict, α_can)
            return (:death, j_star, true)
        end
        return (:death, j_star, false)

    else
        # ===== FIXED DIMENSION MOVE =====
        table_old_target = find_table_for_customer(j_old, state.α_dict)
        table_new_target = find_table_for_customer(j_star, state.α_dict)

        if table_old_target == table_new_target
            # Same table move - just change assignment, keep α
            state_can = GammaClusterShapeMargState(c_can, state.α_dict)

            logpost_current = posterior(model, data, state, priors, log_DDCRP)
            logpost_candidate = posterior(model, data, state_can, priors, log_DDCRP)
            log_α_accept = logpost_candidate - logpost_current

            if log(rand()) < log_α_accept
                state.c[i] = j_star
                return (:fixed, j_star, true)
            end
            return (:fixed, j_star, false)
        end

        # Different tables move
        new_table_depleted = setdiff(table_old_target, S_i)
        new_table_augmented = sort(vcat(table_new_target, S_i))

        # Keep existing α values for the modified tables
        α_depleted = state.α_dict[table_old_target]
        α_augmented = state.α_dict[table_new_target]

        if !isempty(new_table_depleted)
            α_can[new_table_depleted] = α_depleted
        end
        α_can[new_table_augmented] = α_augmented
        delete!(α_can, table_old_target)
        delete!(α_can, table_new_target)

        state_can = GammaClusterShapeMargState(c_can, α_can)
        logpost_current = posterior(model, data, state, priors, log_DDCRP)
        logpost_candidate = posterior(model, data, state_can, priors, log_DDCRP)
        log_α_accept = logpost_candidate - logpost_current

        if log(rand()) < log_α_accept
            state.c[i] = j_star
            empty!(state.α_dict)
            merge!(state.α_dict, α_can)
            return (:fixed, j_star, true)
        end
        return (:fixed, j_star, false)
    end
end

# ============================================================================
# Update Params Orchestration
# ============================================================================

"""
    update_params!(model::GammaClusterShapeMarg, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters.
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
    diagnostics = Vector{Tuple{Symbol, Int, Int, Bool}}()

    # 1. Update shape parameters α (MH on log-scale)
    if should_infer(opts, :α)
        update_α!(model, state, data, priors; prop_sd=get_prop_sd(opts, :α))
    end

    # 2. Update assignments c (RJMCMC)
    if should_infer(opts, :c)
        for i in 1:nobs(data)
            move_type, j_star, accepted = update_c_rjmcmc!(
                model, i, state, data, priors, log_DDCRP, opts
            )
            push!(diagnostics, (move_type, i, j_star, accepted))
        end
    end

    return diagnostics
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
