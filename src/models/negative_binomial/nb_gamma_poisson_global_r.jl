# ============================================================================
# NBGammaPoissonGlobalR - Gamma-Poisson with global r, explicit cluster means
# ============================================================================
#
# Model:
#   y_i | λ_i ~ Poisson(λ_i)
#   λ_i | m_k, r ~ Gamma(r, r/m_k)  for observation i in cluster k
#   m_k ~ InverseGamma(m_a, m_b)    (explicit, sampled)
#   r ~ Gamma(r_a, r_b)             (global dispersion)
#
# Parameters: c (assignments), λ (latent rates), m_k (cluster means), r (global)
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBGammaPoissonGlobalR <: NegativeBinomialModel

Negative Binomial model using Gamma-Poisson parameterisation with global dispersion r.
Cluster means m_k are explicitly maintained and sampled.

Parameters:
- λ: Latent rates (observation-level)
- m_k: Cluster means (cluster-level)
- c: Customer assignments
- r: Global dispersion parameter
"""
struct NBGammaPoissonGlobalR <: NegativeBinomialModel end

# ============================================================================
# State Type
# ============================================================================

"""
    NBGammaPoissonGlobalRState{T<:Real} <: AbstractMCMCState{T}

State for NBGammaPoissonGlobalR model.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `λ::Vector{T}`: Latent rates for each observation
- `m_dict::Dict{Vector{Int}, T}`: Table -> cluster mean mapping
- `r::T`: Global dispersion parameter
"""
mutable struct NBGammaPoissonGlobalRState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ::Vector{T}
    m_dict::Dict{Vector{Int}, T}
    r::T
end

# ============================================================================
# Priors Type
# ============================================================================

"""
    NBGammaPoissonGlobalRPriors{T<:Real} <: AbstractPriors

Prior specification for NBGammaPoissonGlobalR model.

# Fields
- `m_a::T`: InverseGamma shape parameter for cluster mean m
- `m_b::T`: InverseGamma scale parameter for cluster mean m
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NBGammaPoissonGlobalRPriors{T<:Real} <: AbstractPriors
    m_a::T
    m_b::T
    r_a::T
    r_b::T
end

# ============================================================================
# Samples Type
# ============================================================================

"""
    NBGammaPoissonGlobalRSamples{T<:Real} <: AbstractMCMCSamples

MCMC samples container for NBGammaPoissonGlobalR model.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `λ::Matrix{T}`: Latent rates (n_samples x n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `m::Matrix{T}`: Cluster means per observation (n_samples x n_obs)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBGammaPoissonGlobalRSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Matrix{T}
    r::Vector{T}
    m::Matrix{T}
    logpost::Vector{T}
end

# ============================================================================
# Trait Functions
# ============================================================================

has_latent_rates(::NBGammaPoissonGlobalR) = true
has_global_dispersion(::NBGammaPoissonGlobalR) = true
has_cluster_dispersion(::NBGammaPoissonGlobalR) = false
has_cluster_means(::NBGammaPoissonGlobalR) = true
has_cluster_rates(::NBGammaPoissonGlobalR) = false
is_marginalised(::NBGammaPoissonGlobalR) = false

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBGammaPoissonGlobalR, table, state, priors)

Compute log-contribution of a table with explicit cluster mean.
"""
function table_contribution(
    ::NBGammaPoissonGlobalR,
    table::AbstractVector{Int},
    state::NBGammaPoissonGlobalRState,
    priors::NBGammaPoissonGlobalRPriors
)
    n_k = length(table)
    sum_λ = sum(view(state.λ, table))
    m = state.m_dict[sort(table)]
    r = state.r

    norm_term = r * n_k * log(r) - n_k * loggamma(r) - (r * n_k + priors.m_a + 1) * log(m)
    exp_term = -(sum_λ * r + priors.m_b) / m
    prod_term = (r - 1) * sum(log.(view(state.λ, table)))

    return norm_term + exp_term + prod_term
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBGammaPoissonGlobalR, y, state, priors, log_DDCRP)

Compute full log-posterior for unmarginalised Gamma-Poisson NegBin model.
"""
function posterior(
    model::NBGammaPoissonGlobalR,
    y::AbstractVector,
    state::NBGammaPoissonGlobalRState,
    priors::NBGammaPoissonGlobalRPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, priors) for table in keys(state.m_dict)) +
           ddcrp_contribution(state.c, log_DDCRP) +
           likelihood_contribution(y, state.λ)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_λ!(model::NBGammaPoissonGlobalR, i, y, state, priors, tables; prop_sd=0.5)

Update latent rate λ[i] using Metropolis-Hastings with Normal proposal.
"""
function update_λ!(
    model::NBGammaPoissonGlobalR,
    i::Int,
    y::AbstractVector,
    state::NBGammaPoissonGlobalRState,
    priors::NBGammaPoissonGlobalRPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    λ_can = copy(state.λ)
    λ_can[i] = rand(Normal(state.λ[i], prop_sd))

    λ_can[i] <= 0 && return

    table_i = findfirst(x -> i in x, tables)
    state_can = NBGammaPoissonGlobalRState(state.c, λ_can, state.m_dict, state.r)

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
    update_r!(model::NBGammaPoissonGlobalR, state, priors, tables; prop_sd=0.5)

Update global dispersion parameter r using Metropolis-Hastings.
"""
function update_r!(
    model::NBGammaPoissonGlobalR,
    state::NBGammaPoissonGlobalRState,
    priors::NBGammaPoissonGlobalRPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    r_can = rand(Normal(state.r, prop_sd))
    r_can <= 0 && return

    state_can = NBGammaPoissonGlobalRState(state.c, state.λ, state.m_dict, r_can)

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
    update_m!(model::NBGammaPoissonGlobalR, state, priors; prop_sd=0.5)

Update all cluster means using Metropolis-Hastings.
"""
function update_m!(
    model::NBGammaPoissonGlobalR,
    state::NBGammaPoissonGlobalRState,
    priors::NBGammaPoissonGlobalRPriors;
    prop_sd::Float64 = 0.5
)
    for table in keys(state.m_dict)
        update_m_table!(model, table, state, priors; prop_sd=prop_sd)
    end
end

"""
    update_m_table!(model::NBGammaPoissonGlobalR, table, state, priors; prop_sd=0.5)

Update cluster mean for a single table.
"""
function update_m_table!(
    model::NBGammaPoissonGlobalR,
    table::Vector{Int},
    state::NBGammaPoissonGlobalRState,
    priors::NBGammaPoissonGlobalRPriors;
    prop_sd::Float64 = 0.5
)
    m_can = copy(state.m_dict)
    m_can[table] = rand(Normal(state.m_dict[table], prop_sd))

    m_can[table] <= 0 && return

    state_can = NBGammaPoissonGlobalRState(state.c, state.λ, m_can, state.r)

    logpost_current = table_contribution(model, table, state, priors)
    logpost_candidate = table_contribution(model, table, state_can, priors)

    log_accept_ratio = logpost_candidate - logpost_current

    if log(rand()) < log_accept_ratio
        state.m_dict[table] = m_can[table]
    end
end

"""
    update_params!(model::NBGammaPoissonGlobalR, state, y, priors, tables, log_DDCRP, opts)

Update all model parameters (λ, m, r) and customer assignments.
Returns diagnostics information for assignment updates.
"""
function update_params!(
    model::NBGammaPoissonGlobalR,
    state::NBGammaPoissonGlobalRState,
    y::AbstractVector,
    priors::NBGammaPoissonGlobalRPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    diagnostics = Vector{Tuple{Symbol, Int, Int, Bool}}()

    # Update λ
    if should_infer(opts, :λ)
        for i in eachindex(y)
            update_λ!(model, i, y, state, priors, tables; prop_sd=get_prop_sd(opts, :λ))
        end
    end

    # Update m
    if should_infer(opts, :m)
        update_m!(model, state, priors; prop_sd=get_prop_sd(opts, :m))
    end

    # Update r
    if should_infer(opts, :r)
        update_r!(model, state, priors, tables; prop_sd=get_prop_sd(opts, :r))
    end

    # Update customer assignments (this is an unmarginalised model, so uses RJMCMC)
    if should_infer(opts, :c)
        assignment_method = determine_assignment_method(model, opts)
        if assignment_method == :rjmcmc
            for i in eachindex(y)
                move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, y, priors, log_DDCRP, opts)
                push!(diagnostics, (move_type, i, j_star, accepted))
            end
        else
            error("NBGammaPoissonGlobalR is unmarginalised and requires RJMCMC for assignment updates")
        end
    end

    return diagnostics
end

# ============================================================================
# State Initialization
# ============================================================================

"""
    initialise_state(model::NBGammaPoissonGlobalR, y, D, ddcrp_params, priors)

Create initial MCMC state for the model.
"""
function initialise_state(
    ::NBGammaPoissonGlobalR,
    y::AbstractVector,
    D::AbstractMatrix,
    ddcrp_params::DDCRPParams,
    priors::NBGammaPoissonGlobalRPriors
)
    c = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    λ = Float64.(y) .+ 1.0
    r = 1.0
    tables = table_vector(c)
    m_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        m_dict[sort(table)] = mean(view(λ, table))
    end
    return NBGammaPoissonGlobalRState(c, λ, m_dict, r)
end

# ============================================================================
# Sample Allocation and Extraction
# ============================================================================

"""
    allocate_samples(model::NBGammaPoissonGlobalR, n_samples, n)

Allocate storage for MCMC samples.
"""
function allocate_samples(::NBGammaPoissonGlobalR, n_samples::Int, n::Int)
    NBGammaPoissonGlobalRSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # λ
        zeros(n_samples),           # r
        zeros(n_samples, n),        # m (per observation)
        zeros(n_samples)            # logpost
    )
end

"""
    extract_samples!(model::NBGammaPoissonGlobalR, state, samples, iter)

Extract current state into sample storage at iteration iter.
"""
function extract_samples!(
    ::NBGammaPoissonGlobalR,
    state::NBGammaPoissonGlobalRState,
    samples::NBGammaPoissonGlobalRSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    samples.λ[iter, :] = state.λ
    samples.r[iter] = state.r
    samples.m[iter, :] = m_dict_to_samples(1:length(state.c), state.m_dict)
end
