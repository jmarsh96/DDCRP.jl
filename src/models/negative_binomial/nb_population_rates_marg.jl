# ============================================================================
# NBPopulationRatesMarg - NB with population offsets, marginalised cluster rates
# ============================================================================
#
# Model:
#   y_i | γ_k, P_i   ~ Poisson(P_i · γ_k)
#   γ_k               ~ Gamma(r, rate = r/μ)   [E[γ_k] = μ]  (marginalised out)
#   r                 ~ Gamma(r_a, r_b)         (global dispersion)
#
# Conjugate update (used to marginalise γ_k):
#   γ_k | y, c ~ Gamma(r + Σ_{i∈k} y_i,  r/μ + Σ_{i∈k} P_i)
#
# Table contribution (log, after marginalising γ_k):
#   TC_k = Σ_{i∈k} [y_i·log(P_i) − log Γ(y_i+1)]
#          + r·log(r) − r·log(μ) − log Γ(r)
#          + log Γ(r + S_k) − (r + S_k)·log(r/μ + P_sum)
#
#   where S_k = Σ_{i∈k} y_i,  P_sum = Σ_{i∈k} P_i
#
# Parameters: c (assignments), r (global dispersion)
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
Observations follow y_i | γ_k ~ Poisson(P_i · γ_k), where cluster-specific rates
γ_k ~ Gamma(r, r/μ) are marginalised out using Poisson–Gamma conjugacy.

Parameters:
- c: Customer assignments
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
- `r::T`: Global dispersion parameter
"""
mutable struct NBPopulationRatesMargState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    r::T
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBPopulationRatesMargPriors{T<:Real} <: AbstractPriors

Prior specification for NBPopulationRatesMarg model.

# Fields
- `μ::T`: Prior mean for cluster rates γ_k (Gamma(r, r/μ) has mean μ)
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NBPopulationRatesMargPriors{T<:Real} <: AbstractPriors
    μ::T
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
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBPopulationRatesMargSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
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
Cluster rates γ_k are integrated out via Poisson–Gamma conjugacy.

The marginal likelihood of {y_i}_{i∈k} after integrating out γ_k is:
  Σ [y_i·log(P_i) − log Γ(y_i+1)]
  + r·log(r) − r·log(μ) − log Γ(r)
  + log Γ(r + S_k) − (r + S_k)·log(r/μ + P_sum)
"""
function table_contribution(
    ::NBPopulationRatesMarg,
    table::AbstractVector{Int},
    state::NBPopulationRatesMargState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesMargPriors
)
    y = observations(data)
    P = trials(data)
    r = state.r
    n_k = length(table)

    y_k = view(y, table)
    P_k = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))

    S_k   = Float64(sum(y_k))
    P_sum = sum(P_k)

    data_term     = sum(y_k[j] * log(P_k[j]) - loggamma(Float64(y_k[j]) + 1) for j in 1:n_k)
    prior_term    = r * log(r) - r * log(priors.μ) - loggamma(r)
    integral_term = loggamma(r + S_k) - (r + S_k) * log(r / priors.μ + P_sum)

    return data_term + prior_term + integral_term
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
    tables = table_vector(state.c)
    return sum(table_contribution(model, table, state, data, priors) for table in tables) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

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

    state_can = NBPopulationRatesMargState(state.c, r_can)

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

Update all model parameters (r). Assignment updates are handled by update_c!.
"""
function update_params!(
    model::NBPopulationRatesMarg,
    state::NBPopulationRatesMargState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesMargPriors,
    tables::Vector{Vector{Int}},
    ::AbstractMatrix,
    opts::MCMCOptions
)
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
r is initialised to 1.0.
"""
function initialise_state(
    ::NBPopulationRatesMarg,
    data::CountDataWithTrials,
    ddcrp_params::DDCRPParams,
    priors::NBPopulationRatesMargPriors
)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    return NBPopulationRatesMargState(c, 1.0)
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
    samples.r[iter] = state.r
end
