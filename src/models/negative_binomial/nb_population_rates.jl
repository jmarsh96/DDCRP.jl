# ============================================================================
# NBPopulationRates - NB with population offsets, augmented Gamma-Poisson
# ============================================================================
#
# Model (augmented representation):
#   y_i | λ_i, P_i    ~ Poisson(P_i · λ_i)
#   λ_i | γ_k, r      ~ Gamma(r, rate = r/γ_k)   [E[λ_i | γ_k] = γ_k]
#   γ_k               ~ InverseGamma(γ_a, γ_b)
#   r                 ~ Gamma(r_a, r_b)
#
# Marginalising λ_i recovers y_i | γ_k ~ NegBin(r, P_i·γ_k).
#
# Conjugate updates:
#   λ_i | y_i, γ_k, r ~ Gamma(y_i + r,  P_i + r/γ_k)            [exact Gibbs]
#   γ_k | {λ_i}_{i∈k} ~ InverseGamma(n_k·r + γ_a,  r·Λ_k + γ_b) [exact Gibbs]
#   r                  ~ MH random walk
#
# Table contribution (conditional on λ_i and γ_k):
#   TC_k = Σ_{i∈k} [ y_i·log(P_i) − log Γ(y_i+1) ]
#          + Σ_{i∈k} [ (y_i + r − 1)·log(λ_i) − P_i·λ_i ]
#          + n_k·(r·log(r) − log Γ(r))
#          + γ_a·log(γ_b) − log Γ(γ_a)
#          − (n_k·r + γ_a + 1)·log(γ_k) − (r·Λ_k + γ_b)/γ_k
#   where Λ_k = Σ_{i∈k} λ_i
#
# RJMCMC birth proposal for γ_k: conjugate posterior
#   InverseGamma(|S_i|·r + γ_a,  r·Λ_{S_i} + γ_b)
#
# Parameters: c, λ (individual rates), γ_k (cluster rates), r
# Requires: exposure/population data P_i via CountDataWithPopulation
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

# ============================================================================
# Type Definition
# ============================================================================

"""
    NBPopulationRates <: NegativeBinomialModel

Negative Binomial model with population offsets using an augmented Gamma-Poisson
hierarchy. Observations follow y_i | λ_i ~ Poisson(P_i · λ_i), where individual
rates λ_i | γ_k, r ~ Gamma(r, r/γ_k) and cluster rates γ_k ~ InverseGamma(γ_a, γ_b).

Both λ_i and γ_k are maintained explicitly. λ_i and γ_k are updated via exact
conjugate Gibbs; customer assignments via RJMCMC; r via MH.

Requires exposure data P_i via CountDataWithPopulation.
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
- `λ::Vector{T}`: Individual-level latent rates
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
- `γ_a::T`: InverseGamma shape parameter for cluster rates γ_k
- `γ_b::T`: InverseGamma scale parameter for cluster rates γ_k
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
- `c::Matrix{Int}`: Customer assignments (n_samples × n_obs)
- `λ::Matrix{T}`: Individual-level rates (n_samples × n_obs)
- `γ::Matrix{T}`: Cluster rates per observation (n_samples × n_obs)
- `r::Vector{T}`: Global dispersion parameter (n_samples)
- `logpost::Vector{T}`: Log-posterior values (n_samples)
"""
struct NBPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Matrix{T}
    γ::Matrix{T}
    r::Vector{T}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
end

requires_population(::NBPopulationRates) = true

# ============================================================================
# RJMCMC Interface
# ============================================================================

cluster_param_dicts(state::NBPopulationRatesState) = (γ = state.γ_dict,)


"""
    sample_birth_params(model::NBPopulationRates, ::PriorProposal, S_i, state, data, priors)

Sample a new cluster rate γ for a birth move using the conjugate posterior
InverseGamma(|S_i|·r + γ_a, r·Λ_{S_i} + γ_b) conditioned on the current λ values
for the moving set S_i.
"""
function sample_birth_params(
    ::NBPopulationRates,
    ::PriorProposal,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    r    = state.r
    mask = data.missing_mask
    n_Si = count(i -> !mask[i], S_i)
    Λ_Si = sum(state.λ[i] for i in S_i if !mask[i]; init=0.0)

    α_post = n_Si * r + priors.γ_a
    β_post = r * Λ_Si + priors.γ_b
    Q      = InverseGamma(α_post, β_post)
    γ_new  = rand(Q)
    return (γ = γ_new,), logpdf(Q, γ_new)
end

"""
    birth_params_logpdf(model::NBPopulationRates, ::PriorProposal, params_old, S_i, state, data, priors)

Log-density of the birth proposal for a death (reverse) move.
Evaluates the conjugate posterior InverseGamma at the existing γ value.
"""
function birth_params_logpdf(
    ::NBPopulationRates,
    ::PriorProposal,
    params_old::NamedTuple,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    r    = state.r
    mask = data.missing_mask
    n_Si = count(i -> !mask[i], S_i)
    Λ_Si = sum(state.λ[i] for i in S_i if !mask[i]; init=0.0)

    α_post = n_Si * r + priors.γ_a
    β_post = r * Λ_Si + priors.γ_b
    Q      = InverseGamma(α_post, β_post)
    return logpdf(Q, params_old.γ)
end

# --- FixedDistributionProposal ---
# Samples γ_new from a user-specified fixed distribution Q = prop.dists[1],
# e.g. FixedDistributionProposal([Exponential(0.01)]).
function sample_birth_params(
    ::NBPopulationRates,
    prop::FixedDistributionProposal,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    Q     = prop.dists[1]
    γ_new = rand(Q)
    return (γ = γ_new,), logpdf(Q, γ_new)
end

function birth_params_logpdf(
    ::NBPopulationRates,
    prop::FixedDistributionProposal,
    params_old::NamedTuple,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    return logpdf(prop.dists[1], params_old.γ)
end

# --- LogNormalMomentMatch ---
# Proposes γ on the log scale: log(γ) ~ Normal(log(mean(λ_Si)), prop.σ[1]).
# Since E[λ_i | γ_k] = γ_k, the mean of λ values in S_i is a natural moment estimate.
# Falls back to prior mean for empty S_i.
# Jacobian: logpdf_γ = logpdf_log_γ - log(γ).
function sample_birth_params(
    ::NBPopulationRates,
    prop::LogNormalMomentMatch,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    mask     = data.missing_mask
    S_i_obs  = filter(i -> !mask[i], S_i)
    λ_center = isempty(S_i_obs) ? priors.γ_b / max(priors.γ_a - 1.0, 0.01) :
                                   mean(state.λ[j] for j in S_i_obs)
    μ_log    = log(max(λ_center, 1e-8))
    Q        = Normal(μ_log, prop.σ[1])
    log_γ    = rand(Q)
    γ_new    = exp(log_γ)
    return (γ = γ_new,), logpdf(Q, log_γ) - log_γ
end

function birth_params_logpdf(
    ::NBPopulationRates,
    prop::LogNormalMomentMatch,
    params_old::NamedTuple,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    params_old.γ <= 0 && return -Inf
    mask     = data.missing_mask
    S_i_obs  = filter(i -> !mask[i], S_i)
    λ_center = isempty(S_i_obs) ? priors.γ_b / max(priors.γ_a - 1.0, 0.01) :
                                   mean(state.λ[j] for j in S_i_obs)
    μ_log = log(max(λ_center, 1e-8))
    Q     = Normal(μ_log, prop.σ[1])
    return logpdf(Q, log(params_old.γ)) - log(params_old.γ)
end

# --- InverseGammaMomentMatch ---
# Matches an InverseGamma to the empirical moments of existing γ_k values across
# all current clusters. Falls back to the prior when fewer than prop.min_size
# clusters exist or moment matching is degenerate.
function sample_birth_params(
    ::NBPopulationRates,
    prop::InverseGammaMomentMatch,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    γ_vals = collect(values(state.γ_dict))
    if length(γ_vals) >= prop.min_size
        μ_γ  = mean(γ_vals)
        σ²_γ = var(γ_vals)
        if σ²_γ > 0 && μ_γ > 0
            α_ig = 2.0 + μ_γ^2 / σ²_γ
            β_ig = μ_γ * (α_ig - 1.0)
            Q    = InverseGamma(α_ig, β_ig)
            γ_new = rand(Q)
            return (γ = γ_new,), logpdf(Q, γ_new)
        end
    end
    Q = InverseGamma(priors.γ_a, priors.γ_b)
    γ_new = rand(Q)
    return (γ = γ_new,), logpdf(Q, γ_new)
end

function birth_params_logpdf(
    ::NBPopulationRates,
    prop::InverseGammaMomentMatch,
    params_old::NamedTuple,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    γ_vals = collect(values(state.γ_dict))
    if length(γ_vals) >= prop.min_size
        μ_γ  = mean(γ_vals)
        σ²_γ = var(γ_vals)
        if σ²_γ > 0 && μ_γ > 0
            α_ig = 2.0 + μ_γ^2 / σ²_γ
            β_ig = μ_γ * (α_ig - 1.0)
            return logpdf(InverseGamma(α_ig, β_ig), params_old.γ)
        end
    end
    return logpdf(InverseGamma(priors.γ_a, priors.γ_b), params_old.γ)
end

# ============================================================================
# Per-parameter dispatch — required by Resample fixed-dim proposal
# ============================================================================
# sample_birth_param / birth_param_logpdf with Val{:γ} delegate to the
# corresponding plural sample_birth_params / birth_params_logpdf, enabling
# Resample(proposal) as a fixed-dimension proposal for any supported birth proposal.

function sample_birth_param(
    model::NBPopulationRates,
    ::Val{:γ},
    proposal::BirthProposal,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    params, lq = sample_birth_params(model, proposal, S_i, state, data, priors)
    return params.γ, lq
end

function birth_param_logpdf(
    model::NBPopulationRates,
    ::Val{:γ},
    proposal::BirthProposal,
    γ_val,
    S_i::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    return birth_params_logpdf(model, proposal, (γ = γ_val,), S_i, state, data, priors)
end

# ============================================================================
# Fixed-dimension proposals
# ============================================================================

# WeightedMean override: use mean(λ_Si) rather than mean(y_Si) as the summary
# statistic for γ, since E[λ_i | γ_k, r] = γ_k makes the λ mean the natural
# estimate of the cluster rate. The deterministic bijection and Jacobian
# follow the same derivation as the generic WeightedMean.
function fixed_dim_param(
    ::NBPopulationRates,
    ::Val{:γ},
    ::WeightedMean,
    S_i::Vector{Int},
    table_depl::Vector{Int},
    table_aug::Vector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    ::NBPopulationRatesPriors
)
    γ_depl  = state.γ_dict[table_depl]
    γ_aug   = state.γ_dict[table_aug]
    mask    = data.missing_mask

    # Only observed members contribute to the weighted mean update.
    S_i_obs  = filter(i -> !mask[i], S_i)
    n_Si     = length(S_i_obs)
    λ̄_Si     = isempty(S_i_obs) ? 0.0 : mean(state.λ[j] for j in S_i_obs)
    n_depl   = count(i -> !mask[i], table_depl)
    n_aug    = count(i -> !mask[i], table_aug)

    γ_aug_new   = (n_aug * γ_aug + n_Si * λ̄_Si) / (n_aug + n_Si)
    log_jac_aug = log(n_aug) - log(n_aug + n_Si)

    n_remaining = n_depl - n_Si
    if n_remaining > 0
        γ_depl_new = (n_depl * γ_depl - n_Si * λ̄_Si) / n_remaining
        if γ_depl_new <= 0
            return γ_depl_new, γ_aug_new, -Inf
        end
        lpr = log_jac_aug + log(n_depl) - log(n_remaining)
    else
        γ_depl_new = γ_depl   # depleted cluster will be empty; value unused
        lpr        = log_jac_aug
    end

    return γ_depl_new, γ_aug_new, lpr
end

# ============================================================================
# Table Contribution
# ============================================================================

"""
    table_contribution(model::NBPopulationRates, table, state, data, priors)

Compute log-contribution of a table conditional on explicit λ_i and γ_k.
Missing observations have their λ_i integrated out analytically (∫ Gamma dλ_i = 1),
so only observed members contribute.

TC_k = Σ_{i∈k,obs} [ y_i·log(P_i) − log Γ(y_i+1) ]
     + Σ_{i∈k,obs} [ (y_i + r − 1)·log(λ_i) − P_i·λ_i ]
     + n_obs·(r·log(r) − log Γ(r))
     + γ_a·log(γ_b) − log Γ(γ_a)
     − (n_obs·r + γ_a + 1)·log(γ_k) − (r·Λ_obs + γ_b)/γ_k
where n_obs and Λ_obs count observed members only.
"""
function table_contribution(
    ::NBPopulationRates,
    table::AbstractVector{Int},
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors
)
    y  = observations(data)
    P  = population(data)
    γ  = state.γ_dict[sort(table)]
    r  = state.r
    n_k = length(table)

    # γ must be positive (InverseGamma support); guard against temporarily invalid
    # values that can arise during WeightedMean fixed-dim moves before rejection.
    γ <= 0 && return -Inf

    mask = data.missing_mask
    y_k = view(y, table)
    P_k = P isa Int ? fill(Float64(P), n_k) : Float64.(view(P, table))
    λ_k = view(state.λ, table)

    # Missing obs: λ_i integrated out analytically, contributes nothing.
    poisson_const = 0.0
    lambda_terms  = 0.0
    n_obs         = 0
    Λ_obs         = 0.0
    for (j, i) in enumerate(table)
        mask[i] && continue
        n_obs         += 1
        Λ_obs         += λ_k[j]
        poisson_const += Float64(y_k[j]) * log(P_k[j]) - loggamma(Float64(y_k[j]) + 1)
        lambda_terms  += (Float64(y_k[j]) + r - 1) * log(λ_k[j]) - P_k[j] * λ_k[j]
    end
    gamma_norm    = n_obs * (r * log(r) - loggamma(r))
    ig_norm       = priors.γ_a * log(priors.γ_b) - loggamma(priors.γ_a)
    gamma_kernel  = -(n_obs * r + priors.γ_a + 1) * log(γ) - (r * Λ_obs + priors.γ_b) / γ

    return poisson_const + lambda_terms + gamma_norm + ig_norm + gamma_kernel
end

# ============================================================================
# Posterior
# ============================================================================

"""
    posterior(model::NBPopulationRates, data, state, priors, log_DDCRP)

Compute full log-posterior for the augmented NB population rates model.
"""
function posterior(
    model::NBPopulationRates,
    data::CountDataWithPopulation,
    state::NBPopulationRatesState,
    priors::NBPopulationRatesPriors,
    log_DDCRP::AbstractMatrix
)
    return sum(table_contribution(model, sort(table), state, data, priors)
               for table in keys(state.γ_dict)) +
           logpdf(Gamma(priors.r_a, 1 / priors.r_b), state.r) +
           ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    update_λ_gibbs!(model::NBPopulationRates, state, data, priors, tables)

Update individual-level rates λ_i using exact conjugate Gibbs sampling.
Posterior: Gamma(y_i + r,  P_i + r/γ_k) for observation i in cluster k.
"""
function update_λ_gibbs!(
    ::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}}
)
    y    = observations(data)
    P    = population(data)
    mask = data.missing_mask
    r    = state.r
    for table in tables
        γ = state.γ_dict[sort(table)]
        for i in table
            mask[i] && continue  # λ_i integrated out analytically; skip
            P_i    = P isa Int ? Float64(P) : Float64(P[i])
            α_post = Float64(y[i]) + r
            β_post = P_i + r / γ
            state.λ[i] = rand(Gamma(α_post, 1.0 / β_post))
        end
    end
end

"""
    update_γ_ig_rates!(model::NBPopulationRates, state, data, priors, tables)

Update cluster rates γ_k using exact conjugate Gibbs sampling.
Posterior: InverseGamma(n_k·r + γ_a,  r·Λ_k + γ_b) where Λ_k = Σ_{i∈k} λ_i.
"""
function update_γ_ig_rates!(
    ::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}}
)
    r    = state.r
    mask = data.missing_mask
    for table in tables
        key   = sort(table)
        n_obs = count(i -> !mask[i], table)
        Λ_obs = sum(state.λ[i] for i in table if !mask[i]; init=0.0)
        α_post = n_obs * r + priors.γ_a
        β_post = r * Λ_obs + priors.γ_b
        state.γ_dict[key] = rand(InverseGamma(α_post, β_post))
    end
end

"""
    update_r!(model::NBPopulationRates, state, data, priors, tables; prop_sd=0.5)

Update global dispersion r using Metropolis-Hastings with a normal random walk.
"""
function update_r!(
    model::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}};
    prop_sd::Float64 = 0.5
)
    r_can = rand(Normal(state.r, prop_sd))
    r_can <= 0 && return

    state_can = NBPopulationRatesState(state.c, state.λ, state.γ_dict, r_can)

    logpost_current   = sum(table_contribution(model, table, state, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1 / priors.r_b), state.r)
    logpost_candidate = sum(table_contribution(model, table, state_can, data, priors) for table in tables) +
                        logpdf(Gamma(priors.r_a, 1 / priors.r_b), r_can)

    if log(rand()) < logpost_candidate - logpost_current
        state.r = r_can
    end
end

"""
    update_params!(model::NBPopulationRates, state, data, priors, tables, log_DDCRP, opts)

Update all model parameters: λ_i and γ_k via exact Gibbs, r via MH.
Assignment updates are handled by update_c!.
"""
function update_params!(
    model::NBPopulationRates,
    state::NBPopulationRatesState,
    data::CountDataWithPopulation,
    priors::NBPopulationRatesPriors,
    tables::Vector{Vector{Int}},
    ::AbstractMatrix,
    opts::MCMCOptions
)
    if should_infer(opts, :λ)
        update_λ_gibbs!(model, state, data, priors, tables)
    end
    if should_infer(opts, :γ)
        update_γ_ig_rates!(model, state, data, priors, tables)
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

Create initial MCMC state. Assignments drawn from ddCRP prior; λ_i initialised
from smoothed empirical rates; γ_k drawn from conjugate posterior given initial λ;
r initialised to 1.0.
"""
function initialise_state(
    ::NBPopulationRates,
    data::CountDataWithPopulation,
    ddcrp_params::DDCRPParams,
    priors::NBPopulationRatesPriors
)
    y  = observations(data)
    P  = population(data)
    D  = distance_matrix(data)
    n  = length(y)
    r0 = 1.0

    c      = simulate_ddcrp(D; α=ddcrp_params.α, scale=ddcrp_params.scale, decay_fn=ddcrp_params.decay_fn)
    tables = table_vector(c)

    # Initialise λ_i from smoothed empirical rate
    P_vec = P isa Int ? fill(Float64(P), n) : Float64.(P)
    λ0    = [max(Float64(y[i]) / P_vec[i], 0.01) for i in 1:n]

    # Initialise γ_k from conjugate posterior given λ0 (observed members only)
    mask   = data.missing_mask
    γ_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        key    = sort(table)
        n_obs  = count(i -> !mask[i], table)
        Λ_obs  = sum(λ0[i] for i in table if !mask[i]; init=0.0)
        α_post = n_obs * r0 + priors.γ_a
        β_post = r0 * Λ_obs + priors.γ_b
        γ_dict[key] = rand(InverseGamma(α_post, β_post))
    end

    return NBPopulationRatesState(c, λ0, γ_dict, r0)
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
        zeros(n_samples),           # logpost
        zeros(n_samples),           # α_ddcrp
        zeros(n_samples),           # s_ddcrp
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
    samples.c[iter, :]  = state.c
    samples.λ[iter, :]  = state.λ
    samples.r[iter]     = state.r
    for (table, γ_val) in state.γ_dict
        for i in table
            samples.γ[iter, i] = γ_val
        end
    end
end

