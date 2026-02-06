# ============================================================================
# NBMeanDispersionClusterR - Direct NegBin with cluster means and cluster r
# ============================================================================
#
# Model:
#   y_i | m_k, r_k ~ NegBin(r_k, r_k/(r_k + m_k))  for observation i in cluster k
#   m_k ~ InverseGamma(m_a, m_b)                   (explicit, sampled)
#   r_k ~ Gamma(r_a, r_b)                          (cluster-specific dispersion)
#
# This is the mean-dispersion parameterisation of NegBin where:
#   E[y] = m_k
#   Var[y] = m_k + m_k²/r_k
#
# Parameters: c (assignments), m_k (cluster means), r_k (cluster dispersion)
# No latent λ rates - direct NegBin likelihood
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBMeanDispersionClusterR <: NegativeBinomialModel

Negative Binomial model using direct mean-dispersion parameterisation
with cluster-specific dispersion parameters.

Parameters:
- m_k: Cluster means (cluster-level)
- r_k: Cluster dispersion (cluster-level)
- c: Customer assignments
"""
struct NBMeanDispersionClusterR <: NegativeBinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    NBMeanDispersionClusterRState{T<:Real} <: AbstractMCMCState{T}

State for NBMeanDispersionClusterR model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `m_dict::Dict{Vector{Int}, T}`: Table -> cluster mean mapping
- `r_dict::Dict{Vector{Int}, T}`: Table -> cluster dispersion mapping
"""
mutable struct NBMeanDispersionClusterRState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    m_dict::Dict{Vector{Int}, T}
    r_dict::Dict{Vector{Int}, T}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBMeanDispersionClusterRPriors{T<:Real} <: AbstractPriors

Prior specification for NBMeanDispersionClusterR model.

# Fields
- `m_a::T`: InverseGamma shape parameter for cluster mean m
- `m_b::T`: InverseGamma scale parameter for cluster mean m
- `r_a::T`: Gamma shape parameter for cluster dispersion r_k
- `r_b::T`: Gamma rate parameter for cluster dispersion r_k
"""
struct NBMeanDispersionClusterRPriors{T<:Real} <: AbstractPriors
    m_a::T
    m_b::T
    r_a::T
    r_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    NBMeanDispersionClusterRSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for NBMeanDispersionClusterR model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `r::Matrix{T}`: Cluster dispersion per observation (n_samples x n_obs)
- `m::Matrix{T}`: Cluster means per observation (n_samples x n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBMeanDispersionClusterRSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    r::Matrix{T}
    m::Matrix{T}
    logpost::Vector{T}
end


# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::NBMeanDispersionClusterRState) = (m = state.m_dict, r = state.r_dict)
copy_cluster_param_dicts(state::NBMeanDispersionClusterRState) = (m = copy(state.m_dict), r = copy(state.r_dict))

function make_candidate_state(::NBMeanDispersionClusterR, state::NBMeanDispersionClusterRState,
                              c_can::Vector{Int}, params_can::NamedTuple)
    NBMeanDispersionClusterRState(c_can, params_can.m, params_can.r)
end

function commit_params!(state::NBMeanDispersionClusterRState, params_can::NamedTuple)
    empty!(state.m_dict); merge!(state.m_dict, params_can.m)
    empty!(state.r_dict); merge!(state.r_dict, params_can.r)
end

function fixed_dim_params(::NBMeanDispersionClusterR, S_i::Vector{Int},
                          table_old::Vector{Int}, table_new::Vector{Int},
                          state::NBMeanDispersionClusterRState, data::CountData,
                          priors::NBMeanDispersionClusterRPriors, opts::MCMCOptions)
    m_depleted, m_augmented, lpr = compute_fixed_dim_means(
        opts.fixed_dim_mode, S_i, observations(data),
        table_old, state.m_dict[table_old],
        table_new, state.m_dict[table_new], priors)
    r_depleted = state.r_dict[table_old]
    r_augmented = state.r_dict[table_new]
    return (m = m_depleted, r = r_depleted), (m = m_augmented, r = r_augmented), lpr
end

# --- PriorProposal ---
function sample_birth_params(::NBMeanDispersionClusterR, ::PriorProposal,
                             S_i::Vector{Int}, state::NBMeanDispersionClusterRState,
                             data::CountData, priors::NBMeanDispersionClusterRPriors)
    Q_m = InverseGamma(priors.m_a, priors.m_b)
    Q_r = Gamma(priors.r_a, 1/priors.r_b)
    m_new = rand(Q_m)
    r_new = rand(Q_r)
    log_q = logpdf(Q_m, m_new) + logpdf(Q_r, r_new)
    return (m = m_new, r = r_new), log_q
end

function birth_params_logpdf(::NBMeanDispersionClusterR, ::PriorProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::NBMeanDispersionClusterRState, data::CountData,
                             priors::NBMeanDispersionClusterRPriors)
    Q_m = InverseGamma(priors.m_a, priors.m_b)
    Q_r = Gamma(priors.r_a, 1/priors.r_b)
    return logpdf(Q_m, params_old.m) + logpdf(Q_r, params_old.r)
end

# --- NormalMomentMatch ---
function sample_birth_params(::NBMeanDispersionClusterR, prop::NormalMomentMatch,
                             S_i::Vector{Int}, state::NBMeanDispersionClusterRState,
                             data::CountData, priors::NBMeanDispersionClusterRPriors)
    y = observations(data)
    y_Si = view(y, S_i)
    μ_m = max(mean(y_Si), 0.01)
    Q_m = truncated(Normal(μ_m, prop.σ[1]), 0.0, Inf)
    m_new = rand(Q_m)
    Q_r = truncated(Normal(1.0, prop.σ[2]), 0.0, Inf)
    r_new = rand(Q_r)
    log_q = logpdf(Q_m, m_new) + logpdf(Q_r, r_new)
    return (m = m_new, r = r_new), log_q
end

function birth_params_logpdf(::NBMeanDispersionClusterR, prop::NormalMomentMatch,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::NBMeanDispersionClusterRState, data::CountData,
                             priors::NBMeanDispersionClusterRPriors)
    y = observations(data)
    y_Si = view(y, S_i)
    μ_m = max(mean(y_Si), 0.01)
    Q_m = truncated(Normal(μ_m, prop.σ[1]), 0.0, Inf)
    Q_r = truncated(Normal(1.0, prop.σ[2]), 0.0, Inf)
    return logpdf(Q_m, params_old.m) + logpdf(Q_r, params_old.r)
end

# --- LogNormalMomentMatch ---
function sample_birth_params(::NBMeanDispersionClusterR, prop::LogNormalMomentMatch,
                             S_i::Vector{Int}, state::NBMeanDispersionClusterRState,
                             data::CountData, priors::NBMeanDispersionClusterRPriors)
    y = observations(data)
    y_Si = view(y, S_i)
    log_μ_m = log(max(mean(y_Si), 0.01))
    log_m = rand(Normal(log_μ_m, prop.σ[1]))
    m_new = exp(log_m)
    log_r = rand(Normal(0.0, prop.σ[2]))
    r_new = exp(log_r)
    log_q = logpdf(Normal(log_μ_m, prop.σ[1]), log_m) - log_m +
            logpdf(Normal(0.0, prop.σ[2]), log_r) - log_r
    return (m = m_new, r = r_new), log_q
end

function birth_params_logpdf(::NBMeanDispersionClusterR, prop::LogNormalMomentMatch,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::NBMeanDispersionClusterRState, data::CountData,
                             priors::NBMeanDispersionClusterRPriors)
    y = observations(data)
    y_Si = view(y, S_i)
    log_μ_m = log(max(mean(y_Si), 0.01))
    log_m = log(params_old.m)
    log_r = log(params_old.r)
    return logpdf(Normal(log_μ_m, prop.σ[1]), log_m) - log_m +
           logpdf(Normal(0.0, prop.σ[2]), log_r) - log_r
end

# --- FixedDistributionProposal ---
function sample_birth_params(::NBMeanDispersionClusterR, prop::FixedDistributionProposal,
                             S_i::Vector{Int}, state::NBMeanDispersionClusterRState,
                             data::CountData, priors::NBMeanDispersionClusterRPriors)
    m_new = rand(prop.dists[1])
    r_new = rand(prop.dists[2])
    log_q = logpdf(prop.dists[1], m_new) + logpdf(prop.dists[2], r_new)
    return (m = m_new, r = r_new), log_q
end

function birth_params_logpdf(::NBMeanDispersionClusterR, prop::FixedDistributionProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::NBMeanDispersionClusterRState, data::CountData,
                             priors::NBMeanDispersionClusterRPriors)
    return logpdf(prop.dists[1], params_old.m) + logpdf(prop.dists[2], params_old.r)
end

# ============================================================================
# Table Contribution
# ============================================================================

# Note: negbin_logpdf is defined in core/state.jl

"""
    table_contribution(model::NBMeanDispersionClusterR, table, state, data, priors)

Compute log-contribution of a table with direct NegBin likelihood and cluster r.
"""
function table_contribution(
    ::NBMeanDispersionClusterR,
    table::AbstractVector{Int},
    state::NBMeanDispersionClusterRState,
    data::CountData,
    priors::NBMeanDispersionClusterRPriors
)
    y = observations(data)
    key = sort(table)
    m = state.m_dict[key]
    r = state.r_dict[key]

    # NegBin log-likelihood for all observations in table
    log_lik = sum(negbin_logpdf(y[i], m, r) for i in table)

    # Priors
    log_prior_m = logpdf(InverseGamma(priors.m_a, priors.m_b), m)
    log_prior_r = logpdf(Gamma(priors.r_a, 1/priors.r_b), r)

    return log_lik + log_prior_m + log_prior_r
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBMeanDispersionClusterR, data, state, priors, log_DDCRP)

Compute full log-posterior for direct NegBin model with cluster r.
"""
function posterior(
    model::NBMeanDispersionClusterR,
    data::CountData,
    state::NBMeanDispersionClusterRState,
    priors::NBMeanDispersionClusterRPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.m_dict)) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_m!(model::NBMeanDispersionClusterR, state, data, priors; prop_sd=0.5)

Update all cluster means using Metropolis-Hastings.
"""
function update_m!(
    model::NBMeanDispersionClusterR,
    state::NBMeanDispersionClusterRState,
    data::CountData,
    priors::NBMeanDispersionClusterRPriors;
    prop_sd::Float64 = 0.5
)
    for table in keys(state.m_dict)
        update_m_table!(model, table, state, data, priors; prop_sd=prop_sd)
    end
end

"""
    update_m_table!(model::NBMeanDispersionClusterR, table, state, data, priors; prop_sd=0.5)

Update cluster mean for a single table.
"""
function update_m_table!(
    model::NBMeanDispersionClusterR,
    table::Vector{Int},
    state::NBMeanDispersionClusterRState,
    data::CountData,
    priors::NBMeanDispersionClusterRPriors;
    prop_sd::Float64 = 0.5
)
    m_can = copy(state.m_dict)
    m_can[table] = rand(Normal(state.m_dict[table], prop_sd))

    m_can[table] <= 0 && return

    state_can = NBMeanDispersionClusterRState(state.c, m_can, state.r_dict)

    logpost_current = table_contribution(model, table, state, data, priors)
    logpost_candidate = table_contribution(model, table, state_can, data, priors)

    log_accept_ratio = logpost_candidate - logpost_current

    if log(rand()) < log_accept_ratio
        state.m_dict[table] = m_can[table]
    end
end

"""
    update_r!(model::NBMeanDispersionClusterR, state, data, priors, tables; prop_sd=0.5)

Update all cluster dispersion parameters using Metropolis-Hastings.
"""
function update_r!(
    model::NBMeanDispersionClusterR,
    state::NBMeanDispersionClusterRState,
    data::CountData,
    priors::NBMeanDispersionClusterRPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    for table in tables
        update_r_table!(model, table, state, data, priors; prop_sd=prop_sd)
    end
end

"""
    update_r_table!(model::NBMeanDispersionClusterR, table, state, data, priors; prop_sd=0.5)

Update cluster dispersion for a single table.
"""
function update_r_table!(
    model::NBMeanDispersionClusterR,
    table::Vector{Int},
    state::NBMeanDispersionClusterRState,
    data::CountData,
    priors::NBMeanDispersionClusterRPriors;
    prop_sd::Float64 = 0.5
)
    key = sort(table)
    r_can = rand(Normal(state.r_dict[key], prop_sd))
    r_can <= 0 && return

    r_dict_can = copy(state.r_dict)
    r_dict_can[key] = r_can
    state_can = NBMeanDispersionClusterRState(state.c, state.m_dict, r_dict_can)

    logpost_current = table_contribution(model, table, state, data, priors)
    logpost_candidate = table_contribution(model, table, state_can, data, priors)

    log_accept_ratio = logpost_candidate - logpost_current

    if log(rand()) < log_accept_ratio
        state.r_dict[key] = r_can
    end
end

"""
    update_params!(model::NBMeanDispersionClusterR, state, data, priors, tables, log_DDCRP, opts)

Update model parameters (m_k and r_k). Assignment updates are handled
separately by `update_c!` in the main MCMC loop.
"""
function update_params!(
    model::NBMeanDispersionClusterR,
    state::NBMeanDispersionClusterRState,
    data::CountData,
    priors::NBMeanDispersionClusterRPriors,
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
    initialise_state(model::NBMeanDispersionClusterR, data, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::NBMeanDispersionClusterR,
    data::CountData,
    ddcrp_params::DDCRPParams,
    priors::NBMeanDispersionClusterRPriors
)
    y = observations(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)
    m_dict = Dict{Vector{Int}, Float64}()
    r_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        key = sort(table)
        m_dict[key] = max(mean(view(y, table)), 0.1)
        r_dict[key] = 1.0
    end
    return NBMeanDispersionClusterRState(c, m_dict, r_dict)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBMeanDispersionClusterR, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBMeanDispersionClusterR, n_samples::Int, n::Int)
    NBMeanDispersionClusterRSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # r (per observation)
        zeros(n_samples, n),        # m (per observation)
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::NBMeanDispersionClusterR, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBMeanDispersionClusterR,
    state::NBMeanDispersionClusterRState,
    samples::NBMeanDispersionClusterRSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    samples.m[iter, :] = m_dict_to_samples(1:length(state.c), state.m_dict)
    # Store r per observation
    for (table, r_val) in state.r_dict
        for i in table
            samples.r[iter, i] = r_val
        end
    end
end
