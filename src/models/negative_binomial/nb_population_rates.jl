# ============================================================================
# NBPopulationRates - NB with population offsets, explicit cluster rates
# ============================================================================
#
# Model:
#   y_i | λ_i            ~ Poisson(λ_i)
#   λ_i | γ_k, r, P_i   ~ Gamma(r, rate = γ_k / P_i)   [E[λ_i] = r·P_i/γ_k]
#   γ_k                  ~ Gamma(γ_a, γ_b)               (explicit, sampled)
#   r                    ~ Gamma(r_a, r_b)               (global dispersion)
#
# Conjugate updates:
#   λ_i | y_i, γ_k, r, P_i ~ Gamma(y_i + r, scale = P_i/(P_i + γ_k))  [Gibbs]
#   γ_k | λ, c, r          ~ Gamma(γ_a + r·|k|, γ_b + Σ_{i∈k} λ_i/P_i) [Gibbs]
#   r                      ~ MH random walk
#
# Table contribution (conditional on γ_k):
#   TC_k = (r·n_k + γ_a − 1)·log(γ_k)
#          − n_k·logΓ(r)
#          − r·Σ_{i∈k} log(P_i)
#          + (r−1)·Σ_{i∈k} log(λ_i)
#          − (γ_b + Σ_{i∈k} λ_i/P_i)·γ_k
#
# Parameters: c, λ, γ_k (cluster rates), r
# RJMCMC birth proposal: posterior Gamma for γ_k given moving set
# Requires: exposure/population data P_i via CountDataWithTrials
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBPopulationRates <: NegativeBinomialModel

Negative Binomial model with population/exposure offsets and global dispersion r.
Cluster-specific rates γ_k are explicitly maintained and updated via conjugate Gibbs.
Customer assignments are updated via RJMCMC.

Parameters:
- c: Customer assignments
- λ: Latent rates (observation-level, updated via conjugate Gibbs)
- γ_k: Cluster rates (cluster-level, updated via conjugate Gibbs)
- r: Global dispersion parameter (updated via MH)

Requires exposure data P_i via CountDataWithTrials.
"""
struct NBPopulationRates <: NegativeBinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    NBPopulationRatesState{T<:Real} <: AbstractMCMCState{T}

State for NBPopulationRates model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `λ::Vector{T}`: Latent rates for each observation
- `γ_dict::Dict{Vector{Int}, T}`: Table -> cluster rate mapping
- `r::T`: Global dispersion parameter
"""
mutable struct NBPopulationRatesState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ::Vector{T}
    γ_dict::Dict{Vector{Int}, T}
    r::T
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBPopulationRatesPriors{T<:Real} <: AbstractPriors

Prior specification for NBPopulationRates model.

# Fields
- `γ_a::T`: Gamma shape parameter for cluster rate γ_k
- `γ_b::T`: Gamma rate parameter for cluster rate γ_k
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NBPopulationRatesPriors{T<:Real} <: AbstractPriors
    γ_a::T
    γ_b::T
    r_a::T
    r_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    NBPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for NBPopulationRates model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `λ::Matrix{T}`: Latent rates (n_samples x n_obs)
- `γ::Matrix{T}`: Cluster rates per observation (n_samples x n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Matrix{T}
    γ::Matrix{T}
    r::Vector{T}
    logpost::Vector{T}
end

requires_trials(::NBPopulationRates) = true

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::NBPopulationRatesState) = (γ = state.γ_dict,)

"""
    fixed_dim_params(model::NBPopulationRates, S_i, table_old, table_new, state, data, priors, opts)

Fixed-dimensional parameter update when moving set S_i transfers between existing clusters.
Uses NoUpdate strategy: cluster rates are kept at current values.
The acceptance ratio therefore depends only on the change in table contributions.
"""
function fixed_dim_params(
    ::NBPopulationRates,
    S_i::Vector{Int},
    table_old::Vector{Int},
    table_new::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors,
    opts::MCMCOptions
)
    γ_depleted  = state.γ_dict[table_old]
    γ_augmented = state.γ_dict[table_new]
    return (γ = γ_depleted,), (γ = γ_augmented,), 0.0
end

# --- PriorProposal (actually samples from conjugate posterior for moving set) ---

"""
    sample_birth_params(model::NBPopulationRates, ::PriorProposal, S_i, state, data, priors)

Sample a new cluster rate γ for a birth move. Uses the conjugate posterior
Gamma(γ_a + r·|S_i|, γ_b + Σ_{i∈S_i} λ_i/P_i) conditioned on the moving set.
This is an informed proposal that adapts to the current latent rates.
"""
function sample_birth_params(
    ::NBPopulationRates,
    ::PriorProposal,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors
)
    P = trials(data)
    λ = state.λ
    r = state.r
    n_Si = length(S_i)

    P_vec = P isa Int ? fill(Float64(P), n_Si) : Float64.(view(P, S_i))
    sum_λ_over_P = sum(λ[S_i[j]] / P_vec[j] for j in 1:n_Si)

    α_post = priors.γ_a + r * n_Si
    β_post = priors.γ_b + sum_λ_over_P
    Q = Gamma(α_post, 1.0 / β_post)
    γ_new = rand(Q)
    return (γ = γ_new,), logpdf(Q, γ_new)
end

"""
    birth_params_logpdf(model::NBPopulationRates, ::PriorProposal, params_old, S_i, state, data, priors)

Log-density of the birth proposal for a death (reverse) move.
Evaluates the conjugate posterior Gamma at the existing γ value.
"""
function birth_params_logpdf(
    ::NBPopulationRates,
    ::PriorProposal,
    params_old::NamedTuple,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors
)
    P = trials(data)
    λ = state.λ
    r = state.r
    n_Si = length(S_i)

    P_vec = P isa Int ? fill(Float64(P), n_Si) : Float64.(view(P, S_i))
    sum_λ_over_P = sum(λ[S_i[j]] / P_vec[j] for j in 1:n_Si)

    α_post = priors.γ_a + r * n_Si
    β_post = priors.γ_b + sum_λ_over_P
    Q = Gamma(α_post, 1.0 / β_post)
    return logpdf(Q, params_old.γ)
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBPopulationRates, table, state, data, priors)

Compute log-contribution of a table conditional on explicit cluster rate γ_k.
Includes the Gamma(r, rate=γ_k/P_i) log-likelihood for each λ_i and the
Gamma(γ_a, γ_b) log-prior for γ_k (omitting the normalizing constant of the prior).
"""
function table_contribution(
    ::NBPopulationRates,
    table::AbstractVector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors
)
    P = trials(data)
    λ = state.λ
    γ = state.γ_dict[sort(table)]
    r = state.r
    n_k = length(table)

    P_vec = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))
    λ_vec = view(λ, table)

    sum_log_P    = sum(log.(P_vec))
    sum_log_λ    = sum(log.(λ_vec))
    sum_λ_over_P = sum(λ_vec[j] / P_vec[j] for j in 1:n_k)

    # log p(λ_i | γ_k, r, P_i) summed over i∈k
    log_lik = (r * n_k + priors.γ_a - 1) * log(γ) -
              n_k * loggamma(r) -
              r * sum_log_P +
              (r - 1) * sum_log_λ -
              (priors.γ_b + sum_λ_over_P) * γ

    return log_lik
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBPopulationRates, data, state, priors, log_DDCRP)

Compute full log-posterior for the non-marginalised NB population rates model.
"""
function posterior(
    model::NBPopulationRates,
    data::CountDataWithTrials,
    state::NBPopulationRatesState,
    priors::NBPopulationRatesPriors,
    log_DDCRP::AbstractMatrix
)
    y = observations(data)
    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.γ_dict)) +
           ddcrp_contribution(state.c, log_DDCRP) +
           likelihood_contribution(y, state.λ)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_cluster_gamma_rates!(model::NBPopulationRates, state, data, priors, tables)

Update cluster rates γ_k using conjugate Gibbs sampling.
Posterior: Gamma(γ_a + r·|k|, γ_b + Σ_{i∈k} λ_i/P_i)
"""
function update_cluster_gamma_rates!(
    ::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}}
)
    P = trials(data)
    λ = state.λ
    r = state.r
    for table in tables
        key = sort(table)
        n_k = length(table)
        P_vec = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))
        sum_λ_over_P = sum(λ[table[j]] / P_vec[j] for j in 1:n_k)

        α_post = priors.γ_a + r * n_k
        β_post = priors.γ_b + sum_λ_over_P
        state.γ_dict[key] = rand(Gamma(α_post, 1.0 / β_post))
    end
end

"""
    update_λ!(model::NBPopulationRates, i, data, state, priors, tables)

Update latent rate λ[i] via conjugate Gibbs sampling.
Full conditional: Gamma(y_i + r, scale = P_i / (P_i + γ_k))
"""
function update_λ!(
    model::NBPopulationRates,
    i::Int,
    data::CountDataWithTrials,
    state::NBPopulationRatesState,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}}
)
    y = observations(data)
    P = trials(data)
    r = state.r

    table_idx = findfirst(x -> i in x, tables)
    γ_k = state.γ_dict[sort(tables[table_idx])]
    P_i = Float64(P isa Int ? P : P[i])

    shape = Float64(y[i]) + r
    scale = P_i / (P_i + γ_k)
    state.λ[i] = rand(Gamma(shape, scale))
end

"""
    update_r!(model::NBPopulationRates, state, data, priors, tables; prop_sd=0.5)

Update global dispersion parameter r using Metropolis-Hastings.
"""
function update_r!(
    model::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    r_can = rand(Normal(state.r, prop_sd))
    r_can <= 0 && return

    state_can = NBPopulationRatesState(state.c, state.λ, state.γ_dict, r_can)

    logpost_current = sum(table_contribution(model, table, state, data, priors) for table in keys(state.γ_dict)) +
                      logpdf(Gamma(priors.r_a, 1 / priors.r_b), state.r)
    logpost_candidate = sum(table_contribution(model, table, state_can, data, priors) for table in keys(state.γ_dict)) +
                        logpdf(Gamma(priors.r_a, 1 / priors.r_b), r_can)

    if log(rand()) < logpost_candidate - logpost_current
        state.r = r_can
    end
end

"""
    update_params!(model::NBPopulationRates, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters (γ_k via Gibbs, λ via Gibbs, r via MH).
Assignment updates are handled by update_c!.
"""
function update_params!(
    model::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    if should_infer(opts, :γ)
        update_cluster_gamma_rates!(model, state, data, priors, tables)
    end

    if should_infer(opts, :λ)
        for i in 1:nobs(data)
            update_λ!(model, i, data, state, priors, tables)
        end
    end

    if should_infer(opts, :r)
        update_r!(model, state, data, priors, tables; prop_sd=get_prop_sd(opts, :r))
    end
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::NBPopulationRates, data, ddcrp_params, priors)

Create initial MCMC state. Assignments are drawn from the ddCRP prior;
λ is initialised near observed counts; γ_k is initialised from empirical rates.
"""
function initialise_state(
    ::NBPopulationRates,
    data::CountDataWithTrials,
    ddcrp_params::DDCRPParams,
    priors::NBPopulationRatesPriors
)
    y = observations(data)
    P = trials(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    λ = Float64.(y) .+ 1.0
    tables = table_vector(c)

    γ_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        key = sort(table)
        n_k = length(table)
        P_vec = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))
        sum_P = sum(P_vec)
        # Initialise at empirical rate: r·Σ P_i / Σ λ_i ≈ r·P_avg/λ_avg
        # (since E[λ_i] = r·P_i/γ_k, we get γ_k ≈ r·Σ P_i / Σ λ_i)
        sum_λ = sum(view(λ, table))
        γ_dict[key] = sum_P > 0 && sum_λ > 0 ? sum_P / sum_λ : 1.0
    end

    return NBPopulationRatesState(c, λ, γ_dict, 1.0)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBPopulationRates, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBPopulationRates, n_samples::Int, n::Int)
    NBPopulationRatesSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # λ
        zeros(n_samples, n),        # γ (per observation)
        zeros(n_samples),           # r
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::NBPopulationRates, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBPopulationRates,
    state::NBPopulationRatesState,
    samples::NBPopulationRatesSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    samples.λ[iter, :] = state.λ
    samples.r[iter] = state.r
    for (table, γ_val) in state.γ_dict
        for i in table
            samples.γ[iter, i] = γ_val
        end
    end
end
