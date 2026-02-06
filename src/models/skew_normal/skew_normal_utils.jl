# ============================================================================
# Skew Normal Utility Functions
# ============================================================================

using Distributions, SpecialFunctions, Statistics

# ============================================================================
# Core Distribution Functions
# ============================================================================

"""
    skewnormal_logpdf(x::Real, ξ::Real, ω::Real, α::Real)

Compute the log-PDF of the skew normal distribution.

The skew normal PDF is:
    f(x | ξ, ω, α) = (2/ω) × φ((x-ξ)/ω) × Φ(α(x-ξ)/ω)

where φ is the standard normal PDF and Φ is the standard normal CDF.

# Arguments
- `x`: Observation
- `ξ`: Location parameter
- `ω`: Scale parameter (ω > 0)
- `α`: Shape parameter (skewness)
"""
function skewnormal_logpdf(x::Real, ξ::Real, ω::Real, α::Real)
    z = (x - ξ) / ω
    # log(2) + logpdf(Normal(), z) - log(ω) + logcdf(Normal(), α * z)
    # Using StatsFuns for numerical stability would be better, but Distributions has normlogcdf
    log_phi = -0.5 * log(2π) - 0.5 * z^2
    log_Phi = logcdf(Normal(), α * z)
    return log(2) - log(ω) + log_phi + log_Phi
end

"""
    delta_from_alpha(α::Real)

Compute δ = α / √(1 + α²) used in the stochastic representation of skew normal.

The skew normal can be written as:
    X = ξ + ω × (δ × |Z| + √(1-δ²) × ε)

where Z, ε are independent standard normals.
"""
function delta_from_alpha(α::Real)
    return α / sqrt(1 + α^2)
end

"""
    alpha_from_delta(δ::Real)

Inverse of `delta_from_alpha`: compute α from δ.
"""
function alpha_from_delta(δ::Real)
    return δ / sqrt(1 - δ^2)
end

# ============================================================================
# Data Augmentation Functions
# ============================================================================

"""
    sample_h_conditional(x::Real, ξ::Real, ω::Real, α::Real)

Sample latent augmentation variable h from its full conditional distribution.

Given the stochastic representation:
    x = ξ + ω × (δ × h + √(1-δ²) × ε)

where h = |z| with z ~ N(0,1), the conditional distribution of h given x is:
    h | x, ξ, ω, α ~ TruncatedNormal(μ_h, σ_h, 0, ∞)

where:
    μ_h = δ × (x - ξ) / ω
    σ_h = √(1 - δ²)
"""
function sample_h_conditional(x::Real, ξ::Real, ω::Real, α::Real)
    δ = delta_from_alpha(α)
    τ = 1 - δ^2
    μ_h = δ * (x - ξ) / ω
    σ_h = sqrt(τ)
    return rand(truncated(Normal(μ_h, σ_h), 0.0, Inf))
end

"""
    h_conditional_logpdf(h::Real, x::Real, ξ::Real, ω::Real, α::Real)

Log-PDF of the conditional distribution of h given x, ξ, ω, α.
Used in Hastings ratio calculations.
"""
function h_conditional_logpdf(h::Real, x::Real, ξ::Real, ω::Real, α::Real)
    δ = delta_from_alpha(α)
    τ = 1 - δ^2
    μ_h = δ * (x - ξ) / ω
    σ_h = sqrt(τ)
    return logpdf(truncated(Normal(μ_h, σ_h), 0.0, Inf), h)
end

# ============================================================================
# Moment Estimation Functions
# ============================================================================

"""
    estimate_skewness(x::AbstractVector{<:Real})

Compute the sample skewness (Fisher's definition).
"""
function estimate_skewness(x::AbstractVector{<:Real})
    n = length(x)
    n < 3 && return 0.0
    μ = mean(x)
    s = std(x; corrected=true)
    s ≈ 0 && return 0.0
    return sum((xi - μ)^3 for xi in x) / (n * s^3)
end

"""
    alpha_from_skewness(γ₁::Real)

Estimate the shape parameter α from sample skewness γ₁.

The theoretical skewness of a skew normal is:
    γ₁ = ((4 - π)/2) × δ³ / (1 - 2δ²/π)^(3/2)

This function uses an approximate inversion based on the observation that
for moderate skewness, α ≈ sign(γ₁) × |γ₁|^(1/3) × C for some constant C.

For robustness, the result is clipped to [-10, 10].
"""
function alpha_from_skewness(γ₁::Real)
    # Approximate relationship: use a cubic-root scaling
    # This is a rough approximation; exact inversion is more complex
    abs(γ₁) < 1e-10 && return 0.0

    # Empirical approximation based on skew normal moment relationship
    c = 1.5  # Tuning constant
    α_est = sign(γ₁) * abs(γ₁)^(1/3) * c

    # Clip to reasonable range
    return clamp(α_est, -10.0, 10.0)
end

"""
    estimate_skewnormal_params(x::AbstractVector{<:Real})

Estimate skew normal parameters (ξ, ω, α) from data using method of moments.

Returns a named tuple (ξ=..., ω=..., α=...).
"""
function estimate_skewnormal_params(x::AbstractVector{<:Real})
    μ = mean(x)
    σ = std(x; corrected=true)
    γ₁ = estimate_skewness(x)

    α_est = alpha_from_skewness(γ₁)
    δ_est = delta_from_alpha(α_est)

    # From the moments of skew normal:
    # E[X] = ξ + ω × δ × √(2/π)
    # Var[X] = ω² × (1 - 2δ²/π)

    b = sqrt(2 / π)
    var_factor = 1 - 2 * δ_est^2 / π

    if var_factor > 0
        ω_est = sqrt(σ^2 / var_factor)
        ξ_est = μ - ω_est * δ_est * b
    else
        # Fallback for extreme δ
        ω_est = σ
        ξ_est = μ
    end

    return (ξ=ξ_est, ω=max(ω_est, 0.01), α=α_est)
end

# ============================================================================
# Augmented Likelihood Functions
# ============================================================================

"""
    augmented_loglik(x::Real, h::Real, ξ::Real, ω::Real, α::Real)

Log-likelihood contribution from observation x with latent h under the augmented model.

The augmented model is:
    x | h, ξ, ω, α ~ N(ξ + ω × δ × h, ω² × (1 - δ²))

where δ = α / √(1 + α²).
"""
function augmented_loglik(x::Real, h::Real, ξ::Real, ω::Real, α::Real)
    δ = delta_from_alpha(α)
    τ = 1 - δ^2
    μ_x = ξ + ω * δ * h
    σ²_x = ω^2 * τ
    return -0.5 * log(2π) - 0.5 * log(σ²_x) - 0.5 * (x - μ_x)^2 / σ²_x
end

"""
    augmented_prior_h(h::Real)

Log-prior for latent h, which follows a half-normal distribution.
h = |z| where z ~ N(0,1), so h ~ HalfNormal(0, 1).
"""
function augmented_prior_h(h::Real)
    h < 0 && return -Inf
    # HalfNormal(0, 1) = sqrt(2/π) * exp(-h²/2) for h >= 0
    return log(2) - 0.5 * log(2π) - 0.5 * h^2
end
