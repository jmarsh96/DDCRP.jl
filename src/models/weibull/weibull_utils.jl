# ============================================================================
# Weibull Utility Functions
# ============================================================================

using Distributions, SpecialFunctions, Statistics

"""
    weibull_logpdf(y, k, λ)

Log-PDF of the Weibull distribution with shape `k` and rate `λ` (where scale = 1/λ).

    log f(y; k, λ) = log(k) + k⋅log(λ) + (k−1)⋅log(y) − (λy)^k

Returns -Inf if y ≤ 0, k ≤ 0, or λ ≤ 0.
"""
function weibull_logpdf(y::Real, k::Real, λ::Real)
    (y <= 0 || k <= 0 || λ <= 0) && return -Inf
    return log(k) + k * log(λ) + (k - 1) * log(y) - (λ * y)^k
end

"""
    fit_weibull_shape_moments(data)

Estimate the Weibull shape parameter `k` from data using the log-standard deviation method.

For X ~ Weibull(k, θ), log(X) approximately follows a Gumbel distribution with
scale 1/k, so k ≈ π / (√6 ⋅ std(log(data))).

Returns `nothing` if:
- fewer than 2 data points
- any data point is non-positive
- log-standard deviation is zero or near-zero
"""
function fit_weibull_shape_moments(data)
    n = length(data)
    n < 2 && return nothing
    any(x -> x <= 0, data) && return nothing

    log_data = log.(data)
    σ_log = std(log_data)

    σ_log <= 1e-10 && return nothing

    k_est = π / (sqrt(6) * σ_log)
    k_est <= 0 && return nothing

    return k_est
end

"""
    fit_weibull_rate_moments(data, k)

Estimate the Weibull rate parameter `λ` given shape `k` using the method of moments.

For X ~ Weibull(k, 1/λ):  E[X] = Γ(1 + 1/k) / λ
So:  λ = Γ(1 + 1/k) / mean(data)

Returns `nothing` if mean(data) ≤ 0 or k ≤ 0.
"""
function fit_weibull_rate_moments(data, k::Real)
    k <= 0 && return nothing
    μ = mean(data)
    μ <= 0 && return nothing

    λ_est = gamma(1.0 + 1.0 / k) / μ
    λ_est <= 0 && return nothing

    return λ_est
end
