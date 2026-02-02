# ============================================================================
# MCMC State Container Types
# ============================================================================

"""
    NegBinMarginalisedState{T<:Real} <: AbstractMCMCState{T}

State for Negative Binomial model with marginalised cluster means.
Cluster means m are integrated out analytically.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `λ::Vector{T}`: Latent rates for each observation
- `r::T`: Global dispersion parameter
"""
mutable struct NegBinMarginalisedState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ::Vector{T}
    r::T
end

"""
    NegBinUnmarginalisedState{T<:Real} <: AbstractMCMCState{T}

State for Negative Binomial model with explicit cluster means.
Uses Dict mapping sorted table vectors to mean values.

# Fields
- `c::Vector{Int}`: Customer assignments (link representation)
- `λ::Vector{T}`: Latent rates for each observation
- `m_dict::Dict{Vector{Int}, T}`: Table -> cluster mean mapping
- `r::T`: Global dispersion parameter
"""
mutable struct NegBinUnmarginalisedState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ::Vector{T}
    m_dict::Dict{Vector{Int}, T}
    r::T
end

"""
    PoissonState{T<:Real} <: AbstractMCMCState{T}

State for Poisson model with cluster-specific rates.

# Fields
- `c::Vector{Int}`: Customer assignments
- `λ_dict::Dict{Vector{Int}, T}`: Table -> cluster rate mapping
"""
mutable struct PoissonState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    λ_dict::Dict{Vector{Int}, T}
end

"""
    BinomialMarginalisedState <: AbstractMCMCState{Float64}

State for Binomial model with marginalised success probability.
Uses Beta-Binomial conjugacy for closed-form marginal likelihood.

# Fields
- `c::Vector{Int}`: Customer assignments
"""
mutable struct BinomialMarginalisedState <: AbstractMCMCState{Float64}
    c::Vector{Int}
end

"""
    BinomialUnmarginalisedState{T<:Real} <: AbstractMCMCState{T}

State for Binomial model with explicit cluster success probabilities.

# Fields
- `c::Vector{Int}`: Customer assignments
- `p_dict::Dict{Vector{Int}, T}`: Table -> probability mapping
"""
mutable struct BinomialUnmarginalisedState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    p_dict::Dict{Vector{Int}, T}
end

# ============================================================================
# MCMC Samples Container
# ============================================================================

"""
    MCMCSamples{T<:Real} <: AbstractMCMCSamples

Container for MCMC output samples.

# Fields
- `c::Matrix{Int}`: Customer assignments (n_samples x n_obs)
- `λ::Union{Matrix{T}, Nothing}`: Latent rates (if applicable)
- `r::Union{Vector{T}, Nothing}`: Dispersion parameter (if applicable)
- `m::Union{Matrix{T}, Nothing}`: Cluster means per observation (if applicable)
- `logpost::Vector{T}`: Log-posterior values
"""
struct MCMCSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    λ::Union{Matrix{T}, Nothing}
    r::Union{Vector{T}, Nothing}
    m::Union{Matrix{T}, Nothing}
    logpost::Vector{T}
end

# ============================================================================
# State Utility Functions
# ============================================================================

"""
    m_dict_to_samples(y, m_dict)

Convert m_dict (table -> mean) to vector of means per observation.
Each observation gets the mean of its table.
"""
function m_dict_to_samples(y, m_dict)
    n = length(y)
    m_vec = zeros(n)
    for (table, m_val) in m_dict
        for i in table
            m_vec[i] = m_val
        end
    end
    return m_vec
end

"""
    likelihood_contribution(y, λ)

Compute Poisson log-likelihood contribution: sum(y * log(λ) - λ).
Used in augmented models where λ are latent rates.
"""
function likelihood_contribution(y, λ)
    ll = 0.0
    @inbounds for i in eachindex(y)
        ll += y[i] * log(λ[i]) - λ[i]
    end
    return ll
end

# ============================================================================
# Shared Model Utility Functions
# ============================================================================

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

"""
    logbinomial(n, k)

Compute log of binomial coefficient C(n,k) = n! / (k! * (n-k)!).
Uses loggamma for numerical stability.
"""
function logbinomial(n, k)
    return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
end
