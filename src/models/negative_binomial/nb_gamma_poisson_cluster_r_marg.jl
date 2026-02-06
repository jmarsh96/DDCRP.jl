# ============================================================================
# NBGammaPoissonClusterRMarg - Gamma-Poisson with cluster-specific r, marginalised means
# ============================================================================
#
# Model:
#   y_i | λ_i ~ Poisson(λ_i)
#   λ_i | m_k, r_k ~ Gamma(r_k, r_k/m_k)  for observation i in cluster k
#   m_k ~ InverseGamma(m_a, m_b)          (marginalised out analytically)
#   r_k ~ Gamma(r_a, r_b)                 (cluster-specific dispersion)
#
# Parameters: c (assignments), λ (latent rates), r_k (cluster dispersion)
# Marginalised: m_k (cluster means integrated out)
# ============================================================================

using Distributions, SpecialFunctions, Random

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBGammaPoissonClusterRMarg <: NegativeBinomialModel

Negative Binomial model using Gamma-Poisson parameterisation with cluster-specific
dispersion r_k. Cluster means m_k are marginalised out using InverseGamma-Gamma conjugacy.

Parameters:
- λ: Latent rates (observation-level)
- c: Customer assignments
- r_k: Cluster-specific dispersion parameters
"""
struct NBGammaPoissonClusterRMarg <: NegativeBinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    NBGammaPoissonClusterRMargState{T<:Real} <: AbstractMCMCState{T}

State for NBGammaPoissonClusterRMarg model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `λ::Vector{T}`: Latent rates for each observation
- `r_dict::Dict{Vector{Int}, T}`: Table -> cluster dispersion mapping
"""
mutable struct NBGammaPoissonClusterRMargState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ::Vector{T}
    r_dict::Dict{Vector{Int}, T}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBGammaPoissonClusterRMargPriors{T<:Real} <: AbstractPriors

Prior specification for NBGammaPoissonClusterRMarg model.

# Fields
- `m_a::T`: InverseGamma shape parameter for cluster mean m
- `m_b::T`: InverseGamma scale parameter for cluster mean m
- `r_a::T`: Gamma shape parameter for cluster dispersion r_k
- `r_b::T`: Gamma rate parameter for cluster dispersion r_k
"""
struct NBGammaPoissonClusterRMargPriors{T<:Real} <: AbstractPriors
    m_a::T
    m_b::T
    r_a::T
    r_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    NBGammaPoissonClusterRMargSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for NBGammaPoissonClusterRMarg model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `λ::Matrix{T}`: Latent rates (n_samples x n_obs)
- `r::Matrix{T}`: Cluster dispersion per observation (n_samples x n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBGammaPoissonClusterRMargSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Matrix{T}
    r::Matrix{T}
    logpost::Vector{T}
end

# ============================================================================
# Trait Functions
# ============================================================================

is_marginalised(::NBGammaPoissonClusterRMarg) = false

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::NBGammaPoissonClusterRMargState) = (r = state.r_dict,)
copy_cluster_param_dicts(state::NBGammaPoissonClusterRMargState) = (r = copy(state.r_dict),)

function make_candidate_state(::NBGammaPoissonClusterRMarg, state::NBGammaPoissonClusterRMargState,
                              c_can::Vector{Int}, params_can::NamedTuple)
    NBGammaPoissonClusterRMargState(c_can, state.λ, params_can.r)
end

function commit_params!(state::NBGammaPoissonClusterRMargState, params_can::NamedTuple)
    empty!(state.r_dict); merge!(state.r_dict, params_can.r)
end

# --- PriorProposal ---
function sample_birth_params(::NBGammaPoissonClusterRMarg, ::PriorProposal,
                             S_i::Vector{Int}, state::NBGammaPoissonClusterRMargState,
                             data::CountData, priors::NBGammaPoissonClusterRMargPriors)
    Q = Gamma(priors.r_a, 1/priors.r_b)
    r_new = rand(Q)
    return (r = r_new,), logpdf(Q, r_new)
end

function birth_params_logpdf(::NBGammaPoissonClusterRMarg, ::PriorProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::NBGammaPoissonClusterRMargState, data::CountData,
                             priors::NBGammaPoissonClusterRMargPriors)
    return logpdf(Gamma(priors.r_a, 1/priors.r_b), params_old.r)
end

# --- FixedDistributionProposal ---
function sample_birth_params(::NBGammaPoissonClusterRMarg, prop::FixedDistributionProposal,
                             S_i::Vector{Int}, state::NBGammaPoissonClusterRMargState,
                             data::CountData, priors::NBGammaPoissonClusterRMargPriors)
    Q = prop.dists[1]
    r_new = rand(Q)
    return (r = r_new,), logpdf(Q, r_new)
end

function birth_params_logpdf(::NBGammaPoissonClusterRMarg, prop::FixedDistributionProposal,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::NBGammaPoissonClusterRMargState, data::CountData,
                             priors::NBGammaPoissonClusterRMargPriors)
    return logpdf(prop.dists[1], params_old.r)
end

# --- NormalMomentMatch ---
function sample_birth_params(::NBGammaPoissonClusterRMarg, prop::NormalMomentMatch,
                             S_i::Vector{Int}, state::NBGammaPoissonClusterRMargState,
                             data::CountData, priors::NBGammaPoissonClusterRMargPriors)
    λ_Si = view(state.λ, S_i)
    μ_λ = mean(λ_Si)
    v_λ = var(λ_Si; corrected=false)
    r_est = v_λ > 0 ? μ_λ^2 / v_λ : priors.r_a / priors.r_b
    Q = truncated(Normal(r_est, prop.σ[1]), 0.0, Inf)
    r_new = rand(Q)
    return (r = r_new,), logpdf(Q, r_new)
end

function birth_params_logpdf(::NBGammaPoissonClusterRMarg, prop::NormalMomentMatch,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::NBGammaPoissonClusterRMargState, data::CountData,
                             priors::NBGammaPoissonClusterRMargPriors)
    λ_Si = view(state.λ, S_i)
    μ_λ = mean(λ_Si)
    v_λ = var(λ_Si; corrected=false)
    r_est = v_λ > 0 ? μ_λ^2 / v_λ : priors.r_a / priors.r_b
    Q = truncated(Normal(r_est, prop.σ[1]), 0.0, Inf)
    return logpdf(Q, params_old.r)
end

# --- LogNormalMomentMatch ---
function sample_birth_params(::NBGammaPoissonClusterRMarg, prop::LogNormalMomentMatch,
                             S_i::Vector{Int}, state::NBGammaPoissonClusterRMargState,
                             data::CountData, priors::NBGammaPoissonClusterRMargPriors)
    λ_Si = view(state.λ, S_i)
    μ_λ = mean(λ_Si)
    v_λ = var(λ_Si; corrected=false)
    r_est = v_λ > 0 ? μ_λ^2 / v_λ : priors.r_a / priors.r_b
    log_r_est = log(max(r_est, 1e-6))
    Q = Normal(log_r_est, prop.σ[1])
    log_r_new = rand(Q)
    r_new = exp(log_r_new)
    return (r = r_new,), logpdf(Q, log_r_new) - log_r_new
end

function birth_params_logpdf(::NBGammaPoissonClusterRMarg, prop::LogNormalMomentMatch,
                             params_old::NamedTuple, S_i::Vector{Int},
                             state::NBGammaPoissonClusterRMargState, data::CountData,
                             priors::NBGammaPoissonClusterRMargPriors)
    λ_Si = view(state.λ, S_i)
    μ_λ = mean(λ_Si)
    v_λ = var(λ_Si; corrected=false)
    r_est = v_λ > 0 ? μ_λ^2 / v_λ : priors.r_a / priors.r_b
    log_r_est = log(max(r_est, 1e-6))
    Q = Normal(log_r_est, prop.σ[1])
    log_r_old = log(params_old.r)
    return logpdf(Q, log_r_old) - log_r_old
end

# --- Fixed-Dimension Params (uses compute_fixed_dim_means for r) ---
function fixed_dim_params(::NBGammaPoissonClusterRMarg, S_i, table_old, table_new,
                          state::NBGammaPoissonClusterRMargState, data, priors, opts)
    r_A = state.r_dict[table_old]
    r_B = state.r_dict[table_new]
    r_A_new, r_B_new, lpr = compute_fixed_dim_means(
        opts.fixed_dim_mode, S_i, state.λ,
        table_old, r_A, table_new, r_B, priors)
    return (r = r_A_new,), (r = r_B_new,), lpr
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBGammaPoissonClusterRMarg, table, state, priors)

Compute log-contribution of a table with marginalised cluster mean and cluster-specific r.
"""
function table_contribution(
    ::NBGammaPoissonClusterRMarg,
    table::AbstractVector{Int},
    state::NBGammaPoissonClusterRMargState,
    priors::NBGammaPoissonClusterRMargPriors
)
    n = length(table)
    sum_λ = sum(view(state.λ, table))
    r = state.r_dict[sort(table)]

    norm_const = n * (r * log(r) - loggamma(r))
    integral_term = loggamma(n * r + priors.m_a) - (n * r + priors.m_a) * log(r * sum_λ + priors.m_b)
    data_term = (r - 1) * sum(log.(view(state.λ, table)))

    # Prior on r_k
    log_prior_r = logpdf(Gamma(priors.r_a, 1/priors.r_b), r)

    return norm_const + integral_term + data_term + log_prior_r
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBGammaPoissonClusterRMarg, data, state, priors, log_DDCRP)

Compute full log-posterior for Gamma-Poisson NegBin model with cluster r.
"""
function posterior(
    model::NBGammaPoissonClusterRMarg,
    data::CountData,
    state::NBGammaPoissonClusterRMargState,
    priors::NBGammaPoissonClusterRMargPriors,
    log_DDCRP::AbstractMatrix
)
    y = observations(data)
    tables = table_vector(state.c)
    return sum(table_contribution(model, table, state, priors) for table in tables) +
           ddcrp_contribution(state.c, log_DDCRP) +
           likelihood_contribution(y, state.λ)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_λ!(model::NBGammaPoissonClusterRMarg, i, data, state, priors, tables; prop_sd=0.5)

Update latent rate λ[i] using Metropolis-Hastings with Normal proposal.
"""
function update_λ!(
    model::NBGammaPoissonClusterRMarg,
    i::Int,
    data::CountData,
    state::NBGammaPoissonClusterRMargState,
    priors::NBGammaPoissonClusterRMargPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    y = observations(data)
    λ_can = copy(state.λ)
    λ_can[i] = rand(Normal(state.λ[i], prop_sd))

    λ_can[i] <= 0 && return

    table_i = findfirst(x -> i in x, tables)
    state_can = NBGammaPoissonClusterRMargState(state.c, λ_can, state.r_dict)

    logpost_current = likelihood_contribution(y, state.λ) +
                      table_contribution(model, tables[table_i], state, priors)
    logpost_candidate = likelihood_contribution(y, λ_can) +
                        table_contribution(model, tables[table_i], state_can, priors)

    log_accept_ratio = logpost_candidate - logpost_current

    if log(rand()) < log_accept_ratio
        state.λ[i] = λ_can[i]
    end
end

"""
    update_r!(model::NBGammaPoissonClusterRMarg, state, priors, tables; prop_sd=0.5)

Update all cluster dispersion parameters using Metropolis-Hastings.
"""
function update_r!(
    model::NBGammaPoissonClusterRMarg,
    state::NBGammaPoissonClusterRMargState,
    priors::NBGammaPoissonClusterRMargPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    for table in tables
        update_r_table!(model, table, state, priors; prop_sd=prop_sd)
    end
end

"""
    update_r_table!(model::NBGammaPoissonClusterRMarg, table, state, priors; prop_sd=0.5)

Update cluster dispersion for a single table.
"""
function update_r_table!(
    model::NBGammaPoissonClusterRMarg,
    table::Vector{Int},
    state::NBGammaPoissonClusterRMargState,
    priors::NBGammaPoissonClusterRMargPriors;
    prop_sd::Float64 = 0.5
)
    key = sort(table)
    r_can = rand(Normal(state.r_dict[key], prop_sd))
    r_can <= 0 && return

    r_dict_can = copy(state.r_dict)
    r_dict_can[key] = r_can
    state_can = NBGammaPoissonClusterRMargState(state.c, state.λ, r_dict_can)

    logpost_current = table_contribution(model, table, state, priors)
    logpost_candidate = table_contribution(model, table, state_can, priors)

    log_accept_ratio = logpost_candidate - logpost_current

    if log(rand()) < log_accept_ratio
        state.r_dict[key] = r_can
    end
end

"""
    update_params!(model::NBGammaPoissonClusterRMarg, state, data, priors, tables, log_DDCRP, opts)

Update λ and r_k parameters. Assignment updates are handled separately by `update_c!`.
"""
function update_params!(
    model::NBGammaPoissonClusterRMarg,
    state::NBGammaPoissonClusterRMargState,
    data::CountData,
    priors::NBGammaPoissonClusterRMargPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    if should_infer(opts, :λ)
        for i in 1:nobs(data)
            update_λ!(model, i, data, state, priors, tables; prop_sd=get_prop_sd(opts, :λ))
        end
    end

    if should_infer(opts, :r)
        update_r!(model, state, priors, tables; prop_sd=get_prop_sd(opts, :r))
    end
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::NBGammaPoissonClusterRMarg, data, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::NBGammaPoissonClusterRMarg,
    data::CountData,
    ddcrp_params::DDCRPParams,
    priors::NBGammaPoissonClusterRMargPriors
)
    y = observations(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    λ = Float64.(y) .+ 1.0
    tables = table_vector(c)
    r_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        r_dict[sort(table)] = 1.0  # Initial dispersion
    end
    return NBGammaPoissonClusterRMargState(c, λ, r_dict)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBGammaPoissonClusterRMarg, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBGammaPoissonClusterRMarg, n_samples::Int, n::Int)
    NBGammaPoissonClusterRMargSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # λ
        zeros(n_samples, n),        # r (per observation, stores cluster r)
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::NBGammaPoissonClusterRMarg, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBGammaPoissonClusterRMarg,
    state::NBGammaPoissonClusterRMargState,
    samples::NBGammaPoissonClusterRMargSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    samples.λ[iter, :] = state.λ
    # Store r per observation (each obs gets its cluster's r)
    for (table, r_val) in state.r_dict
        for i in table
            samples.r[iter, i] = r_val
        end
    end
end
