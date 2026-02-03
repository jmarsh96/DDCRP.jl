# ============================================================================
# Birth Proposal Functions for RJMCMC
# Sampling and density evaluation for different proposal distributions
# ============================================================================

using Distributions, Statistics

"""
    fit_inverse_gamma_moments(data) -> (α, β) or nothing

Fit InverseGamma(α, β) to data using method of moments.
For X ~ InverseGamma(α, β):
  E[X] = β / (α - 1)  for α > 1
  Var[X] = β² / ((α-1)²(α-2))  for α > 2

Returns nothing if fitting fails (insufficient data, zero variance, invalid params).
"""
function fit_inverse_gamma_moments(data)
    n = length(data)
    n < 2 && return nothing

    μ = mean(data)
    σ² = var(data)

    (σ² <= 0 || μ <= 0) && return nothing

    # Method of moments: α = 2 + μ²/σ², β = μ(α-1)
    α = 2 + μ^2 / σ²
    β = μ * (α - 1)

    # Ensure valid parameters (α > 2 for finite variance, β > 0)
    (α <= 2 || β <= 0) && return nothing

    return (α, β)
end

"""
    compute_proposal_σ(prop::NormalMeanProposal, S_i, λ)

Compute the standard deviation for NormalMeanProposal based on the mode.
Handles edge cases: single-element clusters return NaN from std().
"""
function compute_proposal_σ(prop::NormalMeanProposal, S_i, λ)
    if prop.σ_mode == :fixed
        return prop.σ_fixed
    elseif prop.σ_mode == :empirical
        if length(S_i) < 2
            return max(mean(view(λ, S_i)) * 0.5, prop.σ_fixed)
        end
        σ = std(view(λ, S_i))
        return (isnan(σ) || σ <= 0) ? prop.σ_fixed : max(σ, 0.1)
    elseif prop.σ_mode == :scaled
        if length(S_i) < 2
            return max(mean(view(λ, S_i)) * 0.5 / sqrt(length(S_i)), prop.σ_fixed)
        end
        σ = std(view(λ, S_i))
        return (isnan(σ) || σ <= 0) ? prop.σ_fixed / sqrt(length(S_i)) : max(σ / sqrt(length(S_i)), 0.1)
    else
        error("Unknown σ_mode: $(prop.σ_mode)")
    end
end

"""
    compute_lognormal_σ(prop::LogNormalProposal, log_data)

Compute σ for LogNormalProposal, handling NaN from single-element clusters.
"""
function compute_lognormal_σ(prop::LogNormalProposal, log_data)
    if prop.σ_mode == :fixed
        return prop.σ_fixed
    else  # :empirical
        if length(log_data) < 2
            return prop.σ_fixed
        end
        σ = std(log_data)
        return (isnan(σ) || σ <= 0) ? prop.σ_fixed : max(σ, 0.1)
    end
end

# ============================================================================
# PriorProposal - Sample from prior
# ============================================================================

"""
    sample_proposal(::PriorProposal, S_i, λ, priors) -> (m_new, log_q)

Sample new cluster mean from the prior distribution (InverseGamma).
"""
function sample_proposal(::PriorProposal, S_i, λ, priors)
    Q = InverseGamma(priors.m_a, priors.m_b)
    m_new = rand(Q)
    log_q = logpdf(Q, m_new)
    return m_new, log_q
end

"""
    proposal_logpdf(::PriorProposal, m, S_i, λ, priors) -> Float64

Evaluate prior proposal density at m.
"""
function proposal_logpdf(::PriorProposal, m, S_i, λ, priors)
    return logpdf(InverseGamma(priors.m_a, priors.m_b), m)
end

# ============================================================================
# NormalMeanProposal - Truncated Normal centered at empirical mean
# ============================================================================

"""
    sample_proposal(prop::NormalMeanProposal, S_i, λ, priors) -> (m_new, log_q)

Sample new cluster mean from truncated Normal centered at empirical mean.
"""
function sample_proposal(prop::NormalMeanProposal, S_i, λ, priors)
    μ = mean(view(λ, S_i))
    σ = compute_proposal_σ(prop, S_i, λ)

    Q = truncated(Normal(μ, σ), 0.0, Inf)
    m_new = rand(Q)
    log_q = logpdf(Q, m_new)

    return m_new, log_q
end

"""
    proposal_logpdf(prop::NormalMeanProposal, m, S_i, λ, priors) -> Float64
"""
function proposal_logpdf(prop::NormalMeanProposal, m, S_i, λ, priors)
    μ = mean(view(λ, S_i))
    σ = compute_proposal_σ(prop, S_i, λ)
    return logpdf(truncated(Normal(μ, σ), 0.0, Inf), m)
end

# ============================================================================
# MomentMatchedProposal - InverseGamma fitted via method of moments
# ============================================================================

"""
    sample_proposal(prop::MomentMatchedProposal, S_i, λ, priors) -> (m_new, log_q)

Sample from InverseGamma fitted to data via method of moments.
Falls back to prior for small clusters or failed fitting.
"""
function sample_proposal(prop::MomentMatchedProposal, S_i, λ, priors)
    data = view(λ, S_i)

    if length(S_i) < prop.min_size
        return sample_proposal(PriorProposal(), S_i, λ, priors)
    end

    params = fit_inverse_gamma_moments(data)
    if isnothing(params)
        return sample_proposal(PriorProposal(), S_i, λ, priors)
    end

    α, β = params
    Q = InverseGamma(α, β)
    m_new = rand(Q)
    log_q = logpdf(Q, m_new)

    return m_new, log_q
end

"""
    proposal_logpdf(prop::MomentMatchedProposal, m, S_i, λ, priors) -> Float64
"""
function proposal_logpdf(prop::MomentMatchedProposal, m, S_i, λ, priors)
    data = view(λ, S_i)

    if length(S_i) < prop.min_size
        return proposal_logpdf(PriorProposal(), m, S_i, λ, priors)
    end

    params = fit_inverse_gamma_moments(data)
    if isnothing(params)
        return proposal_logpdf(PriorProposal(), m, S_i, λ, priors)
    end

    α, β = params
    return logpdf(InverseGamma(α, β), m)
end

# ============================================================================
# LogNormalProposal - Sample on log scale
# ============================================================================

"""
    sample_proposal(prop::LogNormalProposal, S_i, λ, priors) -> (m_new, log_q)

Sample log(m) ~ Normal(μ, σ) where μ = mean(log(λ[S_i])).
Returns m_new = exp(log_m) with appropriate Jacobian correction.
"""
function sample_proposal(prop::LogNormalProposal, S_i, λ, priors)
    log_data = log.(view(λ, S_i))
    μ = mean(log_data)
    σ = compute_lognormal_σ(prop, log_data)

    log_m = rand(Normal(μ, σ))
    m_new = exp(log_m)

    # Log density includes Jacobian: p(m) = p(log m) / m
    log_q = logpdf(Normal(μ, σ), log_m) - log_m

    return m_new, log_q
end

"""
    proposal_logpdf(prop::LogNormalProposal, m, S_i, λ, priors) -> Float64
"""
function proposal_logpdf(prop::LogNormalProposal, m, S_i, λ, priors)
    log_data = log.(view(λ, S_i))
    μ = mean(log_data)
    σ = compute_lognormal_σ(prop, log_data)

    log_m = log(m)
    return logpdf(Normal(μ, σ), log_m) - log_m
end



function sample_proposal(Q::UnivariateDistribution, S_i, λ, priors)
    m_new = rand(Q)
    log_q = logpdf(Q, m_new)
    return m_new, log_q
end

function proposal_logpdf(Q::UnivariateDistribution, m, S_i, λ, priors)
    return logpdf(Q, m)
end
