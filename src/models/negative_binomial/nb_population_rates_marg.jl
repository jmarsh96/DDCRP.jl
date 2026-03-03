# ============================================================================
# NBPopulationRatesMarg - NB with population offsets, marginalised cluster rates
# ============================================================================
#
# Model (augmented Gamma-Poisson with γ_k integrated out):
#   y_i | λ_i, P_i    ~ Poisson(P_i · λ_i)
#   λ_i | γ_k, r      ~ Gamma(r, rate = r/γ_k)   [E[λ_i | γ_k] = γ_k]
#   γ_k               ~ InverseGamma(γ_a, γ_b)   (integrated out analytically)
#   r                 ~ Gamma(r_a, r_b)
#
# Marginalising γ_k via InverseGamma-Gamma conjugacy gives the table contribution:
#   ∫ ∏_{i∈k} Gamma(λ_i; r, r/γ_k) · IG(γ_k; γ_a, γ_b) dγ_k
#     ∝ Γ(n_k·r + γ_a) / (r·Λ_k + γ_b)^(n_k·r + γ_a)
#
# Table contribution (λ_i explicit, γ_k integrated out):
#   TC_k = Σ_{i∈k} [ y_i·log(P_i·λ_i) − P_i·λ_i − log Γ(y_i+1) ]
#          + n_k·(r·log(r) − log Γ(r))
#          + (r−1)·Σ_{i∈k} log(λ_i)
#          + log Γ(n_k·r + γ_a) − (n_k·r + γ_a)·log(r·Λ_k + γ_b)
#          + γ_a·log(γ_b) − log Γ(γ_a)
#   where Λ_k = Σ_{i∈k} λ_i
#
# Parameters: c (assignments), λ (individual rates), r (global dispersion)
# Marginalised: γ_k (cluster rates integrated out analytically)
# Requires: exposure/population data P_i via CountDataWithTrials
# ============================================================================

using Distributions, SpecialFunctions, Random

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBPopulationRatesMarg <: NegativeBinomialModel

Negative Binomial model with population offsets using an augmented Gamma-Poisson
hierarchy with cluster rates γ_k integrated out analytically.

Model: y_i | λ_i ~ Poisson(P_i · λ_i), λ_i | γ_k, r ~ Gamma(r, r/γ_k),
γ_k ~ InverseGamma(γ_a, γ_b) (marginalised), r ~ Gamma(r_a, r_b).

Individual rates λ_i are maintained explicitly and updated via MH.
Customer assignments are updated via Gibbs (ConjugateProposal required).

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
- `λ::Vector{T}`: Individual-level latent rates
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
- `γ_a::T`: InverseGamma shape parameter for cluster rates γ_k
- `γ_b::T`: InverseGamma scale parameter for cluster rates γ_k
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
- `c::Matrix{Int}`: Customer assignments (n_samples × n_obs)
- `λ::Matrix{T}`: Individual-level rates (n_samples × n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBPopulationRatesMargSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Matrix{T}
    r::Vector{T}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
end

requires_trials(::NBPopulationRatesMarg) = true

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBPopulationRatesMarg, table, state, data, priors)

Compute log-contribution of a table after analytically marginalising γ_k.

TC_k = Σ_{i∈k} [ y_i·log(P_i·λ_i) − P_i·λ_i − log Γ(y_i+1) ]
     + n_k·(r·log(r) − log Γ(r)) + (r−1)·Σ log(λ_i)
     + log Γ(n_k·r + γ_a) − (n_k·r + γ_a)·log(r·Λ_k + γ_b)
     + γ_a·log(γ_b) − log Γ(γ_a)
"""
function table_contribution(
    ::NBPopulationRatesMarg,
    table::AbstractVector{Int},
    state::NBPopulationRatesMargState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesMargPriors
)
    y   = observations(data)
    P   = trials(data)
    r   = state.r
    n_k = length(table)

    y_k = view(y, table)
    P_k = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))
    λ_k = view(state.λ, table)
    Λ_k = sum(λ_k)

    poisson_terms = sum(Float64(y_k[j]) * log(P_k[j] * λ_k[j]) - P_k[j] * λ_k[j] -
                        loggamma(Float64(y_k[j]) + 1) for j in 1:n_k)
    gamma_norm    = n_k * (r * log(r) - loggamma(r))
    lambda_power  = (r - 1) * sum(log(λ_k[j]) for j in 1:n_k)
    ig_integral   = loggamma(n_k * r + priors.γ_a) - (n_k * r + priors.γ_a) * log(r * Λ_k + priors.γ_b)
    ig_norm       = priors.γ_a * log(priors.γ_b) - loggamma(priors.γ_a)

    return poisson_terms + gamma_norm + lambda_power + ig_integral + ig_norm
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
           logpdf(Gamma(priors.r_a, 1 / priors.r_b), state.r) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_λ!(model::NBPopulationRatesMarg, state, data, priors, tables; prop_sd=0.3)

Update individual-level rates λ_i using Metropolis-Hastings with a log-scale
random walk. The acceptance ratio uses the O(1) delta in table contribution.
"""
function update_λ!(
    ::NBPopulationRatesMarg,
    state::NBPopulationRatesMargState,
    data::CountDataWithTrials,
    priors::NBPopulationRatesMargPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.3
)
    y = observations(data)
    P = trials(data)
    r = state.r

    for table in tables
        n_k = length(table)
        y_k = view(y, table)
        P_k = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))
        λ_k = view(state.λ, table)
        Λ_k = sum(λ_k)

        for (j, i) in enumerate(table)
            λ_old  = state.λ[i]
            λ_can  = exp(log(λ_old) + rand(Normal(0.0, prop_sd)))

            Λ_new = Λ_k - λ_old + λ_can

            # O(1) delta in TC: Poisson + Gamma power + IG integral terms
            ΔTC = Float64(y_k[j]) * log(λ_can / λ_old) - P_k[j] * (λ_can - λ_old) +
                  (r - 1) * log(λ_can / λ_old) -
                  (n_k * r + priors.γ_a) * (log(r * Λ_new + priors.γ_b) -
                                             log(r * Λ_k  + priors.γ_b))

            # Log-scale proposal Jacobian: log(λ_can) - log(λ_old)
            log_α = ΔTC + log(λ_can) - log(λ_old)

            if log(rand()) < log_α
                state.λ[i] = λ_can
                Λ_k = Λ_new
            end
        end
    end
end

"""
    update_r!(model::NBPopulationRatesMarg, state, data, priors, tables; prop_sd=0.5)

Update global dispersion r using Metropolis-Hastings with a normal random walk.
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

    logpost_current   = sum(table_contribution(model, table, state, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1 / priors.r_b), state.r)
    logpost_candidate = sum(table_contribution(model, table, state_can, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1 / priors.r_b), r_can)

    if log(rand()) < logpost_candidate - logpost_current
        state.r = r_can
    end
end

"""
    update_params!(model::NBPopulationRatesMarg, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters: λ_i via MH, r via MH.
Assignment updates are handled by update_c!.
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
    if should_infer(opts, :λ)
        update_λ!(model, state, data, priors, tables; prop_sd=get_prop_sd(opts, :λ))
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

Create initial MCMC state. Assignments drawn from the ddCRP prior; λ_i
initialised from smoothed empirical rates; r initialised to 1.0.
"""
function initialise_state(
    ::NBPopulationRatesMarg,
    data::CountDataWithTrials,
    ddcrp_params::DDCRPParams,
    priors::NBPopulationRatesMargPriors
)
    y   = observations(data)
    P   = trials(data)
    D   = distance_matrix(data)
    n   = length(y)

    c   = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)

    P_vec = P isa Int ? fill(Float64(P), n) : Float64.(P)
    λ0    = [max(Float64(y[i]) / P_vec[i], 0.01) for i in 1:n]

    return NBPopulationRatesMargState(c, λ0, 1.0)
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
        zeros(n_samples),           # logpost
        zeros(n_samples),           # α_ddcrp
        zeros(n_samples),           # s_ddcrp
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
    samples.c[iter, :]  = state.c
    samples.λ[iter, :]  = state.λ
    samples.r[iter]     = state.r
end
