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
# Samples Type
# ============================================================================

"""
    BinomialClusterProbSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for BinomialClusterProb model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `p::Matrix{T}`: Cluster probabilities per observation (n_samples x n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct BinomialClusterProbSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    p::Matrix{T}
    logpost::Vector{T}
end


# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::BinomialClusterProbState) = (p = state.p_dict,)
copy_cluster_param_dicts(state::BinomialClusterProbState) = (p = copy(state.p_dict),)

function make_candidate_state(::BinomialClusterProb, state::BinomialClusterProbState,
                              c_can::Vector{Int}, params_can::NamedTuple)
    BinomialClusterProbState(c_can, params_can.p)
end

function commit_params!(state::BinomialClusterProbState, params_can::NamedTuple)
    empty!(state.p_dict); merge!(state.p_dict, params_can.p)
end

# --- PriorProposal (samples from conjugate posterior) ---
function sample_birth_params(::BinomialClusterProb, ::PriorProposal,
                             S_i::Vector{Int}, state::BinomialClusterProbState,
                             data::CountDataWithTrials, priors::BinomialClusterProbPriors)
    y = observations(data)
    N = trials(data)
    y_Si = view(y, S_i)
    N_Si = N isa Int ? fill(N, length(S_i)) : view(N, S_i)
    S_k = sum(y_Si)
    F_k = sum(N_Si) - S_k
    Q = Beta(priors.p_a + S_k, priors.p_b + F_k)
    p_new = rand(Q)
    return (p = p_new,), logpdf(Q, p_new)
end

function birth_params_logpdf(::BinomialClusterProb, ::PriorProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::BinomialClusterProbState, data::CountDataWithTrials,
                             priors::BinomialClusterProbPriors)
    y = observations(data)
    N = trials(data)
    y_Si = view(y, S_i)
    N_Si = N isa Int ? fill(N, length(S_i)) : view(N, S_i)
    S_k = sum(y_Si)
    F_k = sum(N_Si) - S_k
    Q = Beta(priors.p_a + S_k, priors.p_b + F_k)
    return logpdf(Q, params_old.p)
end

# --- FixedDistributionProposal ---
function sample_birth_params(::BinomialClusterProb, prop::FixedDistributionProposal,
                             S_i::Vector{Int}, state::BinomialClusterProbState,
                             data::CountDataWithTrials, priors::BinomialClusterProbPriors)
    Q = prop.dists[1]
    p_new = rand(Q)
    return (p = p_new,), logpdf(Q, p_new)
end

function birth_params_logpdf(::BinomialClusterProb, prop::FixedDistributionProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::BinomialClusterProbState, data::CountDataWithTrials,
                             priors::BinomialClusterProbPriors)
    return logpdf(prop.dists[1], params_old.p)
end

# ============================================================================
# Table Contribution
# ============================================================================

# Note: logbinomial is defined in core/state.jl

"""
    table_contribution(model::BinomialClusterProb, table, state, data, priors)

Compute log-contribution of a table with explicit cluster probability.
"""
function table_contribution(
    ::BinomialClusterProb,
    table::AbstractVector{Int},
    state::BinomialClusterProbState,
    data::CountDataWithTrials,
    priors::BinomialClusterProbPriors
)
    y = observations(data)
    N = trials(data)
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
    posterior(model::BinomialClusterProb, data, state, priors, log_DDCRP)

Compute full log-posterior for Binomial model with explicit probabilities.
"""
function posterior(
    model::BinomialClusterProb,
    data::CountDataWithTrials,
    state::BinomialClusterProbState,
    priors::BinomialClusterProbPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.p_dict)) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_cluster_probs!(model::BinomialClusterProb, state, data, priors, tables)

Update cluster probabilities using conjugate Gibbs sampling.
Posterior: Beta(p_a + S_k, p_b + F_k)
"""
function update_cluster_probs!(
    ::BinomialClusterProb,
    state::BinomialClusterProbState,
    data::CountDataWithTrials,
    priors::BinomialClusterProbPriors,
    tables::Vector{Vector{Int}}
)
    y = observations(data)
    N = trials(data)
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
    update_params!(model::BinomialClusterProb, state, data, priors, tables, log_DDCRP, opts)

Update cluster probabilities via conjugate Gibbs sampling.
Assignment updates are handled separately by `update_c!`.
"""
function update_params!(
    model::BinomialClusterProb,
    state::BinomialClusterProbState,
    data::CountDataWithTrials,
    priors::BinomialClusterProbPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    update_cluster_probs!(model, state, data, priors, tables)
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::BinomialClusterProb, data, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::BinomialClusterProb,
    data::CountDataWithTrials,
    ddcrp_params::DDCRPParams,
    priors::BinomialClusterProbPriors
)
    D = distance_matrix(data)
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
    BinomialClusterProbSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # p - stores cluster prob per observation
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
    samples::BinomialClusterProbSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    for (table, p_val) in state.p_dict
        for i in table
            samples.p[iter, i] = p_val
        end
    end
end
