# ============================================================================
# Utility Functions for Birth Proposals
# Moment matching and distribution fitting helpers
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
    fit_gamma_shape_moments(data) -> Float64 or nothing

Estimate Gamma shape parameter α via method of moments.
For X ~ Gamma(α, β):
  E[X] = α/β
  Var[X] = α/β²

Therefore: α = E[X]²/Var[X] = μ²/σ²

Returns nothing if fitting fails (insufficient data, zero/negative variance).
"""
function fit_gamma_shape_moments(data)
    n = length(data)
    n < 2 && return nothing

    μ = mean(data)
    σ² = var(data; corrected=false)

    (σ² <= 0 || μ <= 0) && return nothing

    # Method of moments: α = μ²/σ²
    α_est = μ^2 / σ²

    # Ensure valid (positive) shape parameter
    α_est <= 0 && return nothing

    return α_est
end
