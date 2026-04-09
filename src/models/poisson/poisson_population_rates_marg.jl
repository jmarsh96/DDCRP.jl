# ============================================================================
# PoissonPopulationRatesMarg - Poisson with population offsets, marginalised rates
# ============================================================================
#
# Model:
#   y_i | ρ_k, P_i ~ Poisson(P_i * ρ_k)   for observation i in cluster k
#   ρ_k ~ Gamma(ρ_a, ρ_b)                  (marginalised via Gamma-Poisson conjugacy)
#
# Gamma-Poisson conjugacy gives the closed-form table contribution (observed only):
#   TC_k = Σ_{i∈k_obs} [y_i·log(P_i) − loggamma(y_i + 1)]
#          + ρ_a·log(ρ_b) − loggamma(ρ_a)
#          + loggamma(S_k + ρ_a) − (S_k + ρ_a)·log(P_k_total + ρ_b)
#
# where S_k = Σ_{i∈k_obs} y_i, P_k_total = Σ_{i∈k_obs} P_i.
# Missing observations contribute nothing to the likelihood (their integral = 1).
#
# Parameters: c (assignments only)
# Marginalised: ρ_k (cluster rate multipliers integrated out)
# Requires: exposure/population data P_i via CountDataWithPopulation
# ============================================================================

using Distributions, SpecialFunctions, Random

# ============================================================================
# Type Definition
# ============================================================================

"""
    PoissonPopulationRatesMarg <: PoissonModel

Poisson model with population/exposure offsets and cluster rates marginalised out.
Uses Gamma-Poisson conjugacy for a closed-form marginal likelihood.

Missing observations are excluded from the likelihood; their cluster assignments
are updated using only the ddCRP prior.

Parameters:
- c: Customer assignments only

Requires population data P_i for each observation via CountDataWithPopulation.
"""
struct PoissonPopulationRatesMarg <: PoissonModel end

# ============================================================================
# State Type
# ============================================================================

"""
    PoissonPopulationRatesMargState <: AbstractMCMCState{Float64}

State for PoissonPopulationRatesMarg model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
"""
mutable struct PoissonPopulationRatesMargState <: AbstractMCMCState{Float64}
    c::Vector{Int}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    PoissonPopulationRatesMargPriors{T<:Real} <: AbstractPriors

Prior specification for PoissonPopulationRatesMarg model.

# Fields
- `ρ_a::T`: Gamma shape parameter for cluster rate multiplier ρ
- `ρ_b::T`: Gamma rate parameter for cluster rate multiplier ρ
"""
struct PoissonPopulationRatesMargPriors{T<:Real} <: AbstractPriors
    ρ_a::T
    ρ_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    PoissonPopulationRatesMargSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for PoissonPopulationRatesMarg model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples × n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
- `α_ddcrp::Vector{T}`: DDCRP concentration samples (n_samples)
- `s_ddcrp::Vector{T}`: DDCRP decay scale samples (n_samples)
"""
struct PoissonPopulationRatesMargSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
end

requires_population(::PoissonPopulationRatesMarg) = true

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::PoissonPopulationRatesMarg, table, state, data, priors)

Compute log-contribution of a table after analytically marginalising ρ_k.
Only observed (non-missing) members contribute to the likelihood.

Returns 0.0 for tables with no observed members (the Gamma integral over the prior = 1).

TC_k = Σ_{i∈k_obs} [y_i·log(P_i) − loggamma(y_i + 1)]
     + ρ_a·log(ρ_b) − loggamma(ρ_a)
     + loggamma(S_k + ρ_a) − (S_k + ρ_a)·log(P_k_total + ρ_b)

where S_k = Σ_{i∈k_obs} y_i, P_k_total = Σ_{i∈k_obs} P_i.
"""
function table_contribution(
    ::PoissonPopulationRatesMarg,
    table::AbstractVector{Int},
    state::PoissonPopulationRatesMargState,
    data::CountDataWithPopulation,
    priors::PoissonPopulationRatesMargPriors
)
    y = observations(data)
    P = population(data)
    mask = data.missing_mask

    obs_table = filter(i -> !mask[i], table)
    isempty(obs_table) && return 0.0

    n_k = length(obs_table)
    y_k = view(y, obs_table)
    P_k = P isa Real ? fill(Float64(P), n_k) : Float64.(view(P, obs_table))

    S_k       = Float64(sum(y_k))
    P_k_total = sum(P_k)

    # Poisson offset terms: Σ y_i·log(P_i) − Σ loggamma(y_i + 1)
    offset_terms = sum(Float64(y_k[j]) * log(P_k[j]) - loggamma(Float64(y_k[j]) + 1)
                       for j in 1:n_k)

    # Gamma prior normalisation
    prior_norm = priors.ρ_a * log(priors.ρ_b) - loggamma(priors.ρ_a)

    # Gamma-Poisson integral
    integral = loggamma(S_k + priors.ρ_a) - (S_k + priors.ρ_a) * log(P_k_total + priors.ρ_b)

    return offset_terms + prior_norm + integral
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::PoissonPopulationRatesMarg, data, state, priors, log_DDCRP)

Compute full log-posterior for the marginalised Poisson population model.
"""
function posterior(
    model::PoissonPopulationRatesMarg,
    data::CountDataWithPopulation,
    state::PoissonPopulationRatesMargState,
    priors::PoissonPopulationRatesMargPriors,
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
    update_params!(model::PoissonPopulationRatesMarg, state, data, priors, tables, log_DDCRP, opts)

No parameter updates needed — cluster rates are fully marginalised out.
Assignment updates are handled by update_c!.
"""
function update_params!(
    ::PoissonPopulationRatesMarg,
    ::PoissonPopulationRatesMargState,
    ::CountDataWithPopulation,
    ::PoissonPopulationRatesMargPriors,
    ::Vector{Vector{Int}},
    ::AbstractMatrix,
    ::MCMCOptions
)
    # No parameter updates - rates are marginalised out
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::PoissonPopulationRatesMarg, data, ddcrp_params, priors)

Create initial MCMC state. Assignments are drawn from the ddCRP prior.
"""
function initialise_state(
    ::PoissonPopulationRatesMarg,
    data::CountDataWithPopulation,
    ddcrp_params::DDCRPParams,
    priors::PoissonPopulationRatesMargPriors
)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    return PoissonPopulationRatesMargState(c)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::PoissonPopulationRatesMarg, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::PoissonPopulationRatesMarg, n_samples::Int, n::Int)
    PoissonPopulationRatesMargSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples),           # logpost
        zeros(n_samples),           # α_ddcrp
        zeros(n_samples),           # s_ddcrp
    )
end

"""
    extract_samples!(model::PoissonPopulationRatesMarg, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::PoissonPopulationRatesMarg,
    state::PoissonPopulationRatesMargState,
    samples::PoissonPopulationRatesMargSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
end

# ============================================================================
# Posterior Predictive
# ============================================================================

"""
    posterior_predictive(model, samples, data, priors)

Generate posterior predictive draws for missing observations.

For each MCMC sample and each missing index j:
1. Find j's cluster in the sampled assignment vector
2. Compute the conjugate posterior for ρ_k using observed cluster members
3. Draw ρ_k ~ Gamma(S_obs + ρ_a, 1/(P_obs + ρ_b))
4. Draw y_j ~ Poisson(P_j * ρ_k)

Falls back to the prior when no observed data is available for the cluster.

# Returns
- `pred::Matrix{Int}`: shape (n_samples, n_missing), predictive draws
- `missing_indices::Vector{Int}`: which original indices are missing
"""
function posterior_predictive(
    ::PoissonPopulationRatesMarg,
    samples::PoissonPopulationRatesMargSamples,
    data::CountDataWithPopulation,
    priors::PoissonPopulationRatesMargPriors
)
    n_samples = size(samples.c, 1)
    y_obs = observations(data)
    P = population(data)
    mask = data.missing_mask

    missing_indices = findall(mask)
    n_missing = length(missing_indices)
    pred = zeros(Int, n_samples, n_missing)

    for s in 1:n_samples
        c_s = samples.c[s, :]
        tables = table_vector(c_s)
        # Build a lookup: observation index → table index in tables vector
        table_lookup = Vector{Int}(undef, length(c_s))
        for (t, tbl) in enumerate(tables)
            for i in tbl
                table_lookup[i] = t
            end
        end

        for (k, j) in enumerate(missing_indices)
            table = tables[table_lookup[j]]
            obs_members = filter(i -> !mask[i], table)

            if isempty(obs_members)
                ρ_k = rand(Gamma(priors.ρ_a, 1.0 / priors.ρ_b))
            else
                S_obs = Float64(sum(y_obs[i] for i in obs_members))
                P_obs = P isa Real ? Float64(P) * length(obs_members) :
                                     sum(Float64(P[i]) for i in obs_members)
                ρ_k = rand(Gamma(S_obs + priors.ρ_a, 1.0 / (P_obs + priors.ρ_b)))
            end

            P_j = P isa Real ? Float64(P) : Float64(P[j])
            pred[s, k] = rand(Poisson(P_j * ρ_k))
        end
    end

    return pred, missing_indices
end
