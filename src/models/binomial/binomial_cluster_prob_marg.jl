# ============================================================================
# BinomialClusterProbMarg - Binomial with marginalised cluster probabilities
# ============================================================================
#
# Model:
#   y_i | p_k, N_i ~ Binomial(N_i, p_k)    for observation i in cluster k
#   p_k ~ Beta(p_a, p_b)                    (marginalised via Beta-Binomial)
#
# Parameters: c (assignments only)
# Marginalised: p_k (cluster probabilities integrated out)
# ============================================================================

using Distributions, SpecialFunctions, Random, StatsBase

# ============================================================================
# Type Definition
# ============================================================================

"""
    BinomialClusterProbMarg <: BinomialModel

Binomial model with cluster probabilities marginalised out.
Uses Beta-Binomial conjugacy for closed-form marginal likelihood.

Parameters:
- c: Customer assignments only
"""
struct BinomialClusterProbMarg <: BinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    BinomialClusterProbMargState <: AbstractMCMCState{Float64}

State for BinomialClusterProbMarg model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
"""
mutable struct BinomialClusterProbMargState <: AbstractMCMCState{Float64}
    c::Vector{Int}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    BinomialClusterProbMargPriors{T<:Real} <: AbstractPriors

Prior specification for BinomialClusterProbMarg model.

# Fields
- `p_a::T`: Beta α parameter for probability p
- `p_b::T`: Beta β parameter for probability p
"""
struct BinomialClusterProbMargPriors{T<:Real} <: AbstractPriors
    p_a::T
    p_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    BinomialClusterProbMargSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for BinomialClusterProbMarg model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct BinomialClusterProbMargSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    logpost::Vector{T}
end

# ============================================================================
# Trait Functions
# ============================================================================

has_latent_rates(::BinomialClusterProbMarg) = false
has_global_dispersion(::BinomialClusterProbMarg) = false
has_cluster_dispersion(::BinomialClusterProbMarg) = false
has_cluster_means(::BinomialClusterProbMarg) = false
has_cluster_rates(::BinomialClusterProbMarg) = false
has_cluster_probs(::BinomialClusterProbMarg) = false
is_marginalised(::BinomialClusterProbMarg) = true

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::BinomialClusterProbMarg, table, state, y, N, priors)

Compute log-contribution of a table with marginalised cluster probability.
Uses Beta-Binomial conjugacy for closed-form marginal.

# Arguments
- `y`: Vector of successes
- `N`: Number of trials (scalar or vector)
"""
function table_contribution(
    ::BinomialClusterProbMarg,
    table::AbstractVector{Int},
    state::BinomialClusterProbMargState,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    priors::BinomialClusterProbMargPriors
)
    n_k = length(table)
    y_table = view(y, table)

    # Handle scalar or vector N
    N_table = N isa Int ? fill(N, n_k) : view(N, table)

    S_k = sum(y_table)                    # Total successes
    F_k = sum(N_table) - S_k              # Total failures

    # Beta-Binomial marginal likelihood
    # log p(y | α, β) = sum(log C(N_i, y_i)) + logB(S + α, F + β) - logB(α, β)
    log_binom = sum(logbinomial.(N_table, y_table))
    log_beta_ratio = logbeta(S_k + priors.p_a, F_k + priors.p_b) -
                     logbeta(priors.p_a, priors.p_b)

    return log_binom + log_beta_ratio
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::BinomialClusterProbMarg, y, N, state, priors, log_DDCRP)

Compute full log-posterior for marginalised Binomial model.
"""
function posterior(
    model::BinomialClusterProbMarg,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    state::BinomialClusterProbMargState,
    priors::BinomialClusterProbMargPriors,
    log_DDCRP::AbstractMatrix
)
    tables = table_vector(state.c)
    return sum(table_contribution(model, table, state, y, N, priors) for table in tables) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_params!(model::BinomialClusterProbMarg, state, y, N, priors, tables; kwargs...)

No parameters to update - probabilities are marginalised out.
"""
function update_params!(
    ::BinomialClusterProbMarg,
    state::BinomialClusterProbMargState,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    priors::BinomialClusterProbMargPriors,
    tables::Vector{Vector{Int}};
    kwargs...
)
    # No-op: probabilities are marginalised out
    return nothing
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::BinomialClusterProbMarg, y, N, D, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::BinomialClusterProbMarg,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    D::AbstractMatrix,
    ddcrp_params::DDCRPParams,
    priors::BinomialClusterProbMargPriors
)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    return BinomialClusterProbMargState(c)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::BinomialClusterProbMarg, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::BinomialClusterProbMarg, n_samples::Int, n::Int)
    BinomialClusterProbMargSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::BinomialClusterProbMarg, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::BinomialClusterProbMarg,
    state::BinomialClusterProbMargState,
    samples::BinomialClusterProbMargSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
end
