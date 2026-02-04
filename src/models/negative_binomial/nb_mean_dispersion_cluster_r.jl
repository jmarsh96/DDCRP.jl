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
# Trait Functions
# ============================================================================

has_latent_rates(::NBMeanDispersionClusterR) = false
has_global_dispersion(::NBMeanDispersionClusterR) = false
has_cluster_dispersion(::NBMeanDispersionClusterR) = true
has_cluster_means(::NBMeanDispersionClusterR) = true
has_cluster_rates(::NBMeanDispersionClusterR) = false
is_marginalised(::NBMeanDispersionClusterR) = false

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

function sample_proposal(::NBMeanDispersionClusterR, Q_m::UnivariateDistribution, Q_r::UnivariateDistribution)
    m_new = rand(Q_m)
    r_new = rand(Q_r)
    log_q = logpdf(Q_m, m_new) + logpdf(Q_r, r_new)
    return m_new, r_new, log_q
end

function proposal_logpdf(::NBMeanDispersionClusterR, Q_m::UnivariateDistribution, Q_r::UnivariateDistribution, m, r)
    log_q = logpdf(Q_m, m) + logpdf(Q_r, r)
    return log_q
end


function update_c_rjmcmc!(
    model::NBMeanDispersionClusterR,
    i::Int,
    state::NBMeanDispersionClusterRState,
    data::CountData,
    priors::NBMeanDispersionClusterRPriors,
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    birth_prop = build_birth_proposal(opts)
    fixed_dim_mode = opts.fixed_dim_mode

    n = length(state.c)
    j_old = state.c[i]

    S_i = get_moving_set(i, state.c)
    table_Si = find_table_for_customer(i, state.r_dict)
    m_old = state.m_dict[table_Si]
    r_old = state.r_dict[table_Si]
    table_l = setdiff(table_Si, S_i)

    j_star = rand(1:n)

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    r_can = copy(state.r_dict)
    m_can = copy(state.m_dict)
    c_can = copy(state.c)
    c_can[i] = j_star

    if !j_old_in_Si && j_star_in_Si
        # BIRTH
        r_new, m_new, log_q_forward = sample_proposal(model, opts.prop_distributions[:Q_m], opts.prop_distributions[:Q_r])

        r_can[sort(S_i)] = r_new
        r_can[sort(table_l)] = state.r_dict[table_Si]
        delete!(r_can, table_Si)

        m_can[sort(S_i)] = m_new
        m_can[sort(table_l)] = state.m_dict[table_Si]
        delete!(m_can, table_Si)

        lpr = -log_q_forward

        state_can = NBMeanDispersionClusterRState(c_can, r_can, m_can)
        log_α = posterior(model, data, state_can, priors, log_DDCRP) -
                posterior(model, data, state, priors, log_DDCRP) + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.r_dict)
            merge!(state.r_dict, r_can)
            empty!(state.m_dict)
            merge!(state.m_dict, m_can)
            return (:birth, j_star, true)
        end
        return (:birth, j_star, false)

    elseif j_old_in_Si && !j_star_in_Si
        # DEATH
        table_target = find_table_for_customer(j_star, state.r_dict)
        r_target = state.r_dict[table_target]
        m_target = state.m_dict[table_target]

        r_can[sort(vcat(table_Si, table_target))] = r_target
        m_can[sort(vcat(table_Si, table_target))] = m_target

        lpr = proposal_logpdf(model, opts.prop_distributions[:Q_m], opts.prop_distributions[:Q_r], r_old, m_old)

        delete!(r_can, table_Si)
        delete!(r_can, table_target)
        delete!(m_can, table_Si)
        delete!(m_can, table_target)

        state_can = NBMeanDispersionClusterRState(c_can, r_can, m_can)
        log_α = posterior(model, data, state_can, priors, log_DDCRP) -
                posterior(model, data, state, priors, log_DDCRP) + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.r_dict)
            merge!(state.r_dict, r_can)
            empty!(state.m_dict)
            merge!(state.m_dict, m_can)
            return (:death, j_star, true)
        end
        return (:death, j_star, false)

    else
        # FIXED DIMENSION
        table_old_target = find_table_for_customer(j_old, state.r_dict)
        table_new_target = find_table_for_customer(j_star, state.r_dict)

        if table_old_target == table_new_target
            state_can = NBMeanDispersionClusterRState(c_can, state.r_dict, state.m_dict)

            log_α = posterior(model, data, state_can, priors, log_DDCRP) -
                    posterior(model, data, state, priors, log_DDCRP)

            if log(rand()) < log_α
                state.c[i] = j_star
                return (:fixed, j_star, true)
            end
            return (:fixed, j_star, false)
        end

        new_table_depleted = setdiff(table_old_target, S_i)
        new_table_augmented = sort(vcat(table_new_target, S_i))

        r_depleted, r_augmented, lpr = compute_fixed_dim_means(
            fixed_dim_mode, S_i, 1.0,
            table_old_target, state.r_dict[table_old_target],
            table_new_target, state.r_dict[table_new_target],
            priors
        )

        m_depleted, m_augmented, _ = compute_fixed_dim_means(
            fixed_dim_mode, S_i, 1.0,
            table_old_target, state.m_dict[table_old_target],
            table_new_target, state.m_dict[table_new_target],
            priors
        )

        r_can[new_table_depleted] = r_depleted
        r_can[new_table_augmented] = r_augmented
        m_can[new_table_depleted] = m_depleted
        m_can[new_table_augmented] = m_augmented
        delete!(r_can, table_old_target)
        delete!(r_can, table_new_target)
        delete!(m_can, table_old_target)
        delete!(m_can, table_new_target)

        state_can = NBMeanDispersionClusterRState(c_can, r_can, m_can)
        log_α = posterior(model, data, state_can, priors, log_DDCRP) -
                posterior(model, data, state, priors, log_DDCRP) + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.r_dict)
            merge!(state.r_dict, r_can)
            empty!(state.m_dict)
            merge!(state.m_dict, m_can)
            return (:fixed, j_star, true)
        end
        return (:fixed, j_star, false)
    end
end

"""
    update_params!(model::NBMeanDispersionClusterR, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters (m_k and r_k).
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
    diagnostics = Vector{Tuple{Symbol, Int, Int, Bool}}()

    if should_infer(opts, :m)
        update_m!(model, state, data, priors; prop_sd=get_prop_sd(opts, :m))
    end

    if should_infer(opts, :r)
        update_r!(model, state, data, priors, tables; prop_sd=get_prop_sd(opts, :r))
    end

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
