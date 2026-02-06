# ============================================================================
# NBGammaPoissonGlobalRMarg - Gamma-Poisson with global r, marginalised means
# ============================================================================
#
# Model:
#   y_i | λ_i ~ Poisson(λ_i)
#   λ_i | m_k, r ~ Gamma(r, r/m_k)  for observation i in cluster k
#   m_k ~ InverseGamma(m_a, m_b)    (marginalised out analytically)
#   r ~ Gamma(r_a, r_b)             (global dispersion)
#
# Parameters: c (assignments), λ (latent rates), r (global dispersion)
# Marginalised: m_k (cluster means integrated out)
# ============================================================================

using Distributions, SpecialFunctions, Random

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBGammaPoissonGlobalRMarg <: NegativeBinomialModel

Negative Binomial model using Gamma-Poisson parameterisation with global dispersion r.
Cluster means m_k are marginalised out using InverseGamma-Gamma conjugacy.

Parameters:
- λ: Latent rates (observation-level)
- c: Customer assignments
- r: Global dispersion parameter
"""
struct NBGammaPoissonGlobalRMarg <: NegativeBinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    NBGammaPoissonGlobalRMargState{T<:Real} <: AbstractMCMCState{T}

State for NBGammaPoissonGlobalRMarg model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `λ::Vector{T}`: Latent rates for each observation
- `r::T`: Global dispersion parameter
"""
mutable struct NBGammaPoissonGlobalRMargState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ::Vector{T}
    r::T
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBGammaPoissonGlobalRMargPriors{T<:Real} <: AbstractPriors

Prior specification for NBGammaPoissonGlobalRMarg model.

# Fields
- `m_a::T`: InverseGamma shape parameter for cluster mean m
- `m_b::T`: InverseGamma scale parameter for cluster mean m
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NBGammaPoissonGlobalRMargPriors{T<:Real} <: AbstractPriors
    m_a::T
    m_b::T
    r_a::T
    r_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    NBGammaPoissonGlobalRMargSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for NBGammaPoissonGlobalRMarg model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `λ::Matrix{T}`: Latent rates (n_samples x n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBGammaPoissonGlobalRMargSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Matrix{T}
    r::Vector{T}
    logpost::Vector{T}
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBGammaPoissonGlobalRMarg, table, state, priors)

Compute log-contribution of a table with marginalised cluster mean.
Integrates out m using InverseGamma-Gamma conjugacy.
"""
function table_contribution(
    ::NBGammaPoissonGlobalRMarg,
    table::AbstractVector{Int},
    state::NBGammaPoissonGlobalRMargState,
    priors::NBGammaPoissonGlobalRMargPriors
)
    n = length(table)
    sum_λ = sum(view(state.λ, table))
    r = state.r

    norm_const = n * (r * log(r) - loggamma(r))
    integral_term = loggamma(n * r + priors.m_a) - (n * r + priors.m_a) * log(r * sum_λ + priors.m_b)
    data_term = (r - 1) * sum(log.(view(state.λ, table)))

    return norm_const + integral_term + data_term
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBGammaPoissonGlobalRMarg, data, state, priors, log_DDCRP)

Compute full log-posterior for marginalised Gamma-Poisson NegBin model.
"""
function posterior(
    model::NBGammaPoissonGlobalRMarg,
    data::CountData,
    state::NBGammaPoissonGlobalRMargState,
    priors::NBGammaPoissonGlobalRMargPriors,
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
    update_λ!(model::NBGammaPoissonGlobalRMarg, i, data, state, priors, tables; prop_sd=0.5)

Update latent rate λ[i] using Metropolis-Hastings with Normal proposal.
"""
function update_λ!(
    model::NBGammaPoissonGlobalRMarg,
    i::Int,
    data::CountData,
    state::NBGammaPoissonGlobalRMargState,
    priors::NBGammaPoissonGlobalRMargPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    y = observations(data)
    λ_can = copy(state.λ)
    λ_can[i] = rand(Normal(state.λ[i], prop_sd))

    # Reject if negative
    λ_can[i] <= 0 && return

    # Find table containing i
    table_i = findfirst(x -> i in x, tables)
    state_can = NBGammaPoissonGlobalRMargState(state.c, λ_can, state.r)

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
    update_r!(model::NBGammaPoissonGlobalRMarg, state, priors, tables; prop_sd=0.5)

Update global dispersion parameter r using Metropolis-Hastings.
"""
function update_r!(
    model::NBGammaPoissonGlobalRMarg,
    state::NBGammaPoissonGlobalRMargState,
    priors::NBGammaPoissonGlobalRMargPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    r_can = rand(Normal(state.r, prop_sd))
    r_can <= 0 && return

    state_can = NBGammaPoissonGlobalRMargState(state.c, state.λ, r_can)

    logpost_current = sum(table_contribution(model, table, state, priors) for table in tables) +
                      logpdf(Gamma(priors.r_a, 1/priors.r_b), state.r)
    logpost_candidate = sum(table_contribution(model, table, state_can, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1/priors.r_b), r_can)

    log_accept_ratio = logpost_candidate - logpost_current

    if log(rand()) < log_accept_ratio
        state.r = r_can
    end
end

"""
    update_params!(model::NBGammaPoissonGlobalRMarg, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters (λ, r) and customer assignments.
Returns diagnostics information for assignment updates.
"""
function update_params!(
    model::NBGammaPoissonGlobalRMarg,
    state::NBGammaPoissonGlobalRMargState,
    data::CountData,
    priors::NBGammaPoissonGlobalRMargPriors,
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
    # Assignment updates handled by update_c!
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::NBGammaPoissonGlobalRMarg, data, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::NBGammaPoissonGlobalRMarg,
    data::CountData,
    ddcrp_params::DDCRPParams,
    priors::NBGammaPoissonGlobalRMargPriors
)
    y = observations(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    λ = Float64.(y) .+ 1.0  # Initialize λ near observed counts
    r = 1.0  # Initial dispersion
    return NBGammaPoissonGlobalRMargState(c, λ, r)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBGammaPoissonGlobalRMarg, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBGammaPoissonGlobalRMarg, n_samples::Int, n::Int)
    NBGammaPoissonGlobalRMargSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # λ
        zeros(n_samples),           # r
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::NBGammaPoissonGlobalRMarg, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBGammaPoissonGlobalRMarg,
    state::NBGammaPoissonGlobalRMargState,
    samples::NBGammaPoissonGlobalRMargSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    samples.λ[iter, :] = state.λ
    samples.r[iter] = state.r
end
