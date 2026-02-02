# ============================================================================
# BinomialClusterProb - Binomial with explicit cluster probabilities
# ============================================================================
#
# Model:
#   y_i | p_k, N_i ~ Binomial(N_i, p_k)    for observation i in cluster k
#   p_k ~ Beta(p_a, p_b)                    (explicit, sampled via conjugacy)
#
# Parameters: c (assignments), p_k (cluster probabilities)
# ============================================================================

using Distributions, SpecialFunctions, Random

# ============================================================================
# Type Definition
# ============================================================================

"""
    BinomialClusterProb <: BinomialModel

Binomial model with explicit cluster-specific success probabilities.
Probabilities p_k are maintained and updated via conjugate Gibbs sampling.

Parameters:
- c: Customer assignments
- p_k: Cluster probabilities (cluster-level)
"""
struct BinomialClusterProb <: BinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    BinomialClusterProbState{T<:Real} <: AbstractMCMCState{T}

State for BinomialClusterProb model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `p_dict::Dict{Vector{Int}, T}`: Table -> cluster probability mapping
"""
mutable struct BinomialClusterProbState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    p_dict::Dict{Vector{Int}, T}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    BinomialClusterProbPriors{T<:Real} <: AbstractPriors

Prior specification for BinomialClusterProb model.

# Fields
- `p_a::T`: Beta α parameter for probability p
- `p_b::T`: Beta β parameter for probability p
"""
struct BinomialClusterProbPriors{T<:Real} <: AbstractPriors
    p_a::T
    p_b::T
end

# ============================================================================
# Trait Functions
# ============================================================================

has_latent_rates(::BinomialClusterProb) = false
has_global_dispersion(::BinomialClusterProb) = false
has_cluster_dispersion(::BinomialClusterProb) = false
has_cluster_means(::BinomialClusterProb) = false
has_cluster_rates(::BinomialClusterProb) = false
has_cluster_probs(::BinomialClusterProb) = true
is_marginalised(::BinomialClusterProb) = false

# ============================================================================
# Table Contribution
# ============================================================================

# Note: logbinomial is defined in core/state.jl

"""
    table_contribution(model::BinomialClusterProb, table, state, y, N, priors)

Compute log-contribution of a table with explicit cluster probability.

# Arguments
- `y`: Vector of successes
- `N`: Number of trials (scalar or vector)
"""
function table_contribution(
    ::BinomialClusterProb,
    table::AbstractVector{Int},
    state::BinomialClusterProbState,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    priors::BinomialClusterProbPriors
)
    p = state.p_dict[sort(table)]
    n_k = length(table)
    y_table = view(y, table)
    N_table = N isa Int ? fill(N, n_k) : view(N, table)

    S_k = sum(y_table)
    F_k = sum(N_table) - S_k

    # Binomial likelihood + Beta prior
    log_binom = sum(logbinomial.(N_table, y_table))
    log_lik = S_k * log(p) + F_k * log(1 - p)
    log_prior = (priors.p_a - 1) * log(p) + (priors.p_b - 1) * log(1 - p)

    return log_binom + log_lik + log_prior
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::BinomialClusterProb, y, N, state, priors, log_DDCRP)

Compute full log-posterior for Binomial model with explicit probabilities.
"""
function posterior(
    model::BinomialClusterProb,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    state::BinomialClusterProbState,
    priors::BinomialClusterProbPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, y, N, priors)
               for table in keys(state.p_dict)) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_cluster_probs!(model::BinomialClusterProb, state, y, N, priors, tables)

Update cluster probabilities using conjugate Gibbs sampling.
Posterior: Beta(p_a + S_k, p_b + F_k)
"""
function update_cluster_probs!(
    ::BinomialClusterProb,
    state::BinomialClusterProbState,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    priors::BinomialClusterProbPriors,
    tables::Vector{Vector{Int}}
)
    for table in tables
        key = sort(table)
        n_k = length(table)
        y_table = view(y, table)
        N_table = N isa Int ? fill(N, n_k) : view(N, table)

        S_k = sum(y_table)
        F_k = sum(N_table) - S_k

        # Conjugate posterior: Beta(α + S_k, β + F_k)
        α_post = priors.p_a + S_k
        β_post = priors.p_b + F_k

        state.p_dict[key] = rand(Beta(α_post, β_post))
    end
end

"""
    update_params!(model::BinomialClusterProb, state, y, N, priors, tables; kwargs...)

Update all model parameters.
"""
function update_params!(
    model::BinomialClusterProb,
    state::BinomialClusterProbState,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    priors::BinomialClusterProbPriors,
    tables::Vector{Vector{Int}};
    kwargs...
)
    update_cluster_probs!(model, state, y, N, priors, tables)
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::BinomialClusterProb, y, N, D, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::BinomialClusterProb,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    D::AbstractMatrix,
    ddcrp_params::DDCRPParams,
    priors::BinomialClusterProbPriors
)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)

    p_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        # Initialize at prior mean
        p_dict[sort(table)] = priors.p_a / (priors.p_a + priors.p_b)
    end

    return BinomialClusterProbState(c, p_dict)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::BinomialClusterProb, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::BinomialClusterProb, n_samples::Int, n::Int)
    MCMCSamples(
        zeros(Int, n_samples, n),   # c
        nothing,                    # λ - not used
        nothing,                    # r - not used
        zeros(n_samples, n),        # m - stores cluster prob per observation
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::BinomialClusterProb, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::BinomialClusterProb,
    state::BinomialClusterProbState,
    samples::MCMCSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    if !isnothing(samples.m)
        for (table, p_val) in state.p_dict
            for i in table
                samples.m[iter, i] = p_val
            end
        end
    end
end
