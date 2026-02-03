# ============================================================================
# Shared Utilities for Negative Binomial Models
# ============================================================================

"""
    likelihood_contribution(y, λ)

Compute Poisson log-likelihood contribution: sum(y * log(λ) - λ).
Used in Gamma-Poisson models where λ are latent observation-level rates.

This function is used by NB models that augment the likelihood with
latent observation-level Poisson rates (NBGammaPoissonGlobalRMarg,
NBGammaPoissonGlobalR, NBGammaPoissonClusterRMarg).
"""
function likelihood_contribution(y::AbstractVector, λ::AbstractVector)
    ll = 0.0
    @inbounds for i in eachindex(y)
        ll += y[i] * log(λ[i]) - λ[i]
    end
    return ll
end

"""
    negbin_logpdf(y, m, r)

Log-PDF of NegBin(r, p) where p = r/(r+m), so E[Y] = m.
Using the parameterisation: P(Y=y) = C(y+r-1, y) * p^r * (1-p)^y

This is the mean-dispersion parameterisation where:
- m is the mean
- r is the dispersion (larger r = less overdispersion)
"""
function negbin_logpdf(y::Real, m::Real, r::Real)
    p = r / (r + m)
    return loggamma(y + r) - loggamma(r) - loggamma(y + 1) +
           r * log(p) + y * log(1 - p)
end
