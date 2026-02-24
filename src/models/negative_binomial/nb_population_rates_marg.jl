# ============================================================================
# NBPopulationRatesMarg - NB with population offsets, marginalised cluster rates
# ============================================================================
#
# Model:
#   y_i | λ_i            ~ Poisson(λ_i)
#   λ_i | γ_k, r, P_i   ~ Gamma(r, rate = γ_k / P_i)   [E[λ_i] = r·P_i/γ_k]
#   γ_k                  ~ Gamma(γ_a, γ_b)               (marginalised out)
#   r                    ~ Gamma(r_a, r_b)               (global dispersion)
#
# Conjugate update (used to marginalise γ_k):
#   γ_k | λ, c, r ~ Gamma(γ_a + r·|k|, γ_b + Σ_{i∈k} λ_i/P_i)
#
# Table contribution (log, after marginalising γ_k):
#   TC_k = n_k·(r·log r − logΓ(r))
#          − r·Σ_{i∈k} log(P_i)
#          + (r−1)·Σ_{i∈k} log(λ_i)
#          + logΓ(n_k·r + γ_a)
#          − (n_k·r + γ_a)·log(γ_b + Σ_{i∈k} λ_i/P_i)
#
# Parameters: c (assignments), λ (latent rates), r (global dispersion)
# Marginalised: γ_k (cluster rates integrated out analytically)
# Requires: exposure/population data P_i via CountDataWithTrials
# ============================================================================

using Distributions, SpecialFunctions, Random

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBPopulationRatesMarg <: NegativeBinomialModel

Negative Binomial model with population/exposure offsets and global dispersion r.
Cluster-specific rates γ_k are marginalised out using Gamma-Gamma conjugacy.

Parameters:
- c: Customer assignments
- λ: Latent rates (observation-level)
- r: Global dispersion parameter

Requires exposure data P_i via CountDataWithTrials.
"""
struct NBPopulationRatesMarg <: NegativeBinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    NBPopulationRatesMargState{T<:Real} <: AbstractMCMCState{T}

State for NBPopulationRatesMarg model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `λ::Vector{T}`: Latent rates for each observation
- `r::T`: Global dispersion parameter
"""
mutable struct NBPopulationRatesMargState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ::Vector{T}
    r::T
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBPopulationRatesMargPriors{T<:Real} <: AbstractPriors

Prior specification for NBPopulationRatesMarg model.

# Fields
- `γ_a::T`: Gamma shape parameter for cluster rate γ_k
- `γ_b::T`: Gamma rate parameter for cluster rate γ_k
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NBPopulationRatesMargPriors{T<:Real} <: AbstractPriors
    γ_a::T
    γ_b::T
    r_a::T
    r_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    NBPopulationRatesMargSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for NBPopulationRatesMarg model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `λ::Matrix{T}`: Latent rates (n_samples x n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBPopulationRatesMargSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Matrix{T}
    r::Vector{T}
    logpost::Vector{T}
end

requires_trials(::NBPopulationRatesMarg) = true

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBPopulationRatesMarg, table, state, data, priors)

Compute log-contribution of a table with population-adjusted NB likelihood.
Cluster rates γ_k are integrated out via Gamma-Gamma conjugacy.

The integration yields:
  ∫ Π_{i∈k} Gamma(λ_i; r, rate=γ_k/P_i) · Gamma(γ_k; γ_a, γ_b) dγ_k
  = n_k·(r·log r − logΓ(r)) − r·Σlog(P_i) + (r−1)·Σlog(λ_i)
    + logΓ(n_k·r + γ_a) − (n_k·r + γ_a)·log(γ_b + Σλ_i/P_i)
"""
function table_contribution(
    ::NBPopulationRatesMarg,
    table::AbstractVector{Int},
    state::NBPopulationRatesMargState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesMargPriors
)
    P = trials(data)
    λ = state.λ
    r = state.r
    n_k = length(table)

    P_vec = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))
    λ_vec = view(λ, table)

    sum_log_P    = sum(log.(P_vec))
    sum_log_λ    = sum(log.(λ_vec))
    sum_λ_over_P = sum(λ_vec[j] / P_vec[j] for j in 1:n_k)

    norm_const     = n_k * (r * log(r) - loggamma(r))
    log_P_term     = -r * sum_log_P
    data_term      = (r - 1) * sum_log_λ
    integral_shape = n_k * r + priors.γ_a
    integral_rate  = priors.γ_b + sum_λ_over_P
    integral_term  = loggamma(integral_shape) - integral_shape * log(integral_rate)

    return norm_const + log_P_term + data_term + integral_term
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBPopulationRatesMarg, data, state, priors, log_DDCRP)

Compute full log-posterior for the marginalised NB population rates model.
"""
function posterior(
    model::NBPopulationRatesMarg,
    data::CountDataWithTrials,
    state::NBPopulationRatesMargState,
    priors::NBPopulationRatesMargPriors,
    log_DDCRP::AbstractMatrix
)
    y = observations(data)
    tables = table_vector(state.c)
    return sum(table_contribution(model, table, state, data, priors) for table in tables) +
           ddcrp_contribution(state.c, log_DDCRP) +
           likelihood_contribution(y, state.λ)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_λ!(model::NBPopulationRatesMarg, i, data, state, priors, tables; prop_sd=0.5)

Update latent rate λ[i] using Metropolis-Hastings with Normal proposal.
Log-target includes Poisson likelihood and the marginalised table contribution.
"""
function update_λ!(
    model::NBPopulationRatesMarg,
    i::Int,
    data::CountDataWithTrials,
    state::NBPopulationRatesMargState,
    priors::NBPopulationRatesMargPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    y = observations(data)
    λ_can = copy(state.λ)
    λ_can[i] = rand(Normal(state.λ[i], prop_sd))

    λ_can[i] <= 0 && return

    table_idx = findfirst(x -> i in x, tables)
    state_can = NBPopulationRatesMargState(state.c, λ_can, state.r)

    logpost_current = likelihood_contribution(y, state.λ) +
                      table_contribution(model, tables[table_idx], state, data, priors)
    logpost_candidate = likelihood_contribution(y, λ_can) +
                        table_contribution(model, tables[table_idx], state_can, data, priors)

    if log(rand()) < logpost_candidate - logpost_current
        state.λ[i] = λ_can[i]
    end
end

"""
    update_r!(model::NBPopulationRatesMarg, state, data, priors, tables; prop_sd=0.5)

Update global dispersion parameter r using Metropolis-Hastings.
Proposal is a Normal random walk; r > 0 is enforced by rejection.
"""
function update_r!(
    model::NBPopulationRatesMarg,
    state::NBPopulationRatesMargState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesMargPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    r_can = rand(Normal(state.r, prop_sd))
    r_can <= 0 && return

    state_can = NBPopulationRatesMargState(state.c, state.λ, r_can)

    logpost_current = sum(table_contribution(model, table, state, data, priors) for table in tables) +
                      logpdf(Gamma(priors.r_a, 1 / priors.r_b), state.r)
    logpost_candidate = sum(table_contribution(model, table, state_can, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1 / priors.r_b), r_can)

    if log(rand()) < logpost_candidate - logpost_current
        state.r = r_can
    end
end

"""
    update_params!(model::NBPopulationRatesMarg, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters (λ, r). Assignment updates are handled by update_c!.
"""
function update_params!(
    model::NBPopulationRatesMarg,
    state::NBPopulationRatesMargState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesMargPriors,
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
        update_r!(model, state, data, priors, tables; prop_sd=get_prop_sd(opts, :r))
    end
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::NBPopulationRatesMarg, data, ddcrp_params, priors)

Create initial MCMC state. Assignments are drawn from the ddCRP prior;
λ is initialised near the observed counts (y + 1).
"""
function initialise_state(
    ::NBPopulationRatesMarg,
    data::CountDataWithTrials,
    ddcrp_params::DDCRPParams,
    priors::NBPopulationRatesMargPriors
)
    y = observations(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    λ = Float64.(y) .+ 1.0
    r = 1.0
    return NBPopulationRatesMargState(c, λ, r)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBPopulationRatesMarg, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBPopulationRatesMarg, n_samples::Int, n::Int)
    NBPopulationRatesMargSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # λ
        zeros(n_samples),           # r
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::NBPopulationRatesMarg, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBPopulationRatesMarg,
    state::NBPopulationRatesMargState,
    samples::NBPopulationRatesMargSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    samples.λ[iter, :] = state.λ
    samples.r[iter] = state.r
end
