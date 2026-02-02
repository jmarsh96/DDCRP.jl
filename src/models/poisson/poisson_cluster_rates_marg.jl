# ============================================================================
# PoissonClusterRatesMarg - Poisson with marginalised cluster rates
# ============================================================================
#
# Model:
#   y_i | λ_k ~ Poisson(λ_k)    for observation i in cluster k
#   λ_k ~ Gamma(λ_a, λ_b)       (marginalised out via Gamma-Poisson conjugacy)
#
# Parameters: c (assignments only)
# Marginalised: λ_k (cluster rates integrated out)
# ============================================================================

using Distributions, SpecialFunctions, Random

# ============================================================================
# Type Definition
# ============================================================================

"""
    PoissonClusterRatesMarg <: PoissonModel

Poisson model with cluster rates marginalised out.
Uses Gamma-Poisson conjugacy for closed-form marginal likelihood.

Parameters:
- c: Customer assignments only
"""
struct PoissonClusterRatesMarg <: PoissonModel end

# ============================================================================
# State Type
# ============================================================================

"""
    PoissonClusterRatesMargState <: AbstractMCMCState{Float64}

State for PoissonClusterRatesMarg model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
"""
mutable struct PoissonClusterRatesMargState <: AbstractMCMCState{Float64}
    c::Vector{Int}
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    PoissonClusterRatesMargPriors{T<:Real} <: AbstractPriors

Prior specification for PoissonClusterRatesMarg model.

# Fields
- `λ_a::T`: Gamma shape parameter for rate λ
- `λ_b::T`: Gamma rate parameter for rate λ
"""
struct PoissonClusterRatesMargPriors{T<:Real} <: AbstractPriors
    λ_a::T
    λ_b::T
end

# ============================================================================
# Trait Functions
# ============================================================================

has_latent_rates(::PoissonClusterRatesMarg) = false
has_global_dispersion(::PoissonClusterRatesMarg) = false
has_cluster_dispersion(::PoissonClusterRatesMarg) = false
has_cluster_means(::PoissonClusterRatesMarg) = false
has_cluster_rates(::PoissonClusterRatesMarg) = false
is_marginalised(::PoissonClusterRatesMarg) = true

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::PoissonClusterRatesMarg, table, state, y, priors)

Compute log-contribution of a table with marginalised cluster rate.
Uses Gamma-Poisson conjugacy for closed-form marginal.
"""
function table_contribution(
    ::PoissonClusterRatesMarg,
    table::AbstractVector{Int},
    state::PoissonClusterRatesMargState,
    y::AbstractVector,
    priors::PoissonClusterRatesMargPriors
)
    n_k = length(table)
    S_k = sum(view(y, table))

    # Gamma-Poisson marginal likelihood
    # p(y_k | α, β) = Γ(S_k + α) / (Γ(α) * prod(y_ki!)) * β^α / (n_k + β)^(S_k + α)
    log_contrib = loggamma(S_k + priors.λ_a) - loggamma(priors.λ_a)
    log_contrib += priors.λ_a * log(priors.λ_b) - (S_k + priors.λ_a) * log(n_k + priors.λ_b)
    log_contrib -= sum(loggamma.(view(y, table) .+ 1))  # -log(y!)

    return log_contrib
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::PoissonClusterRatesMarg, y, state, priors, log_DDCRP)

Compute full log-posterior for marginalised Poisson model.
"""
function posterior(
    model::PoissonClusterRatesMarg,
    y::AbstractVector,
    state::PoissonClusterRatesMargState,
    priors::PoissonClusterRatesMargPriors,
    log_DDCRP::AbstractMatrix
)
    tables = table_vector(state.c)
    return sum(table_contribution(model, table, state, y, priors) for table in tables) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_params!(model::PoissonClusterRatesMarg, state, y, priors, tables; kwargs...)

No parameters to update - rates are marginalised out.
"""
function update_params!(
    ::PoissonClusterRatesMarg,
    state::PoissonClusterRatesMargState,
    y::AbstractVector,
    priors::PoissonClusterRatesMargPriors,
    tables::Vector{Vector{Int}};
    kwargs...
)
    # No-op: rates are marginalised out
    return nothing
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::PoissonClusterRatesMarg, y, D, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::PoissonClusterRatesMarg,
    y::AbstractVector,
    D::AbstractMatrix,
    ddcrp_params::DDCRPParams,
    priors::PoissonClusterRatesMargPriors
)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    return PoissonClusterRatesMargState(c)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::PoissonClusterRatesMarg, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::PoissonClusterRatesMarg, n_samples::Int, n::Int)
    MCMCSamples(
        zeros(Int, n_samples, n),   # c
        nothing,                    # λ - not used
        nothing,                    # r - not used
        nothing,                    # m - rates marginalised
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::PoissonClusterRatesMarg, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::PoissonClusterRatesMarg,
    state::PoissonClusterRatesMargState,
    samples::MCMCSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
end
