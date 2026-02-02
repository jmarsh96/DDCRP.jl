# ============================================================================
# Prior Specification Types
# ============================================================================

"""
    NegBinPriors{T<:Real} <: AbstractPriors

Prior specification for Negative Binomial model.

# Fields
- `m_a::T`: InverseGamma shape parameter for cluster mean m
- `m_b::T`: InverseGamma scale parameter for cluster mean m
- `r_a::T`: Gamma shape parameter for dispersion r
- `r_b::T`: Gamma rate parameter for dispersion r
"""
struct NegBinPriors{T<:Real} <: AbstractPriors
    m_a::T
    m_b::T
    r_a::T
    r_b::T
end

# Convenience constructor from NamedTuple
function NegBinPriors(nt::NamedTuple)
    NegBinPriors(nt.m_a, nt.m_b, nt.r_a, nt.r_b)
end

"""
    PoissonPriors{T<:Real} <: AbstractPriors

Prior specification for Poisson model.

# Fields
- `λ_a::T`: Gamma shape parameter for rate λ
- `λ_b::T`: Gamma rate parameter for rate λ
"""
struct PoissonPriors{T<:Real} <: AbstractPriors
    λ_a::T
    λ_b::T
end

"""
    BinomialPriors{T<:Real} <: AbstractPriors

Prior specification for Binomial model.

# Fields
- `p_a::T`: Beta α parameter for success probability p
- `p_b::T`: Beta β parameter for success probability p
- `N::Int`: Number of trials (can be observation-specific in data)
"""
struct BinomialPriors{T<:Real} <: AbstractPriors
    p_a::T
    p_b::T
end

"""
    DDCRPParams{T<:Real}

DDCRP hyperparameters (shared across models).

# Fields
- `α::T`: Concentration parameter (self-link probability)
- `scale::T`: Distance decay scale parameter
- `decay_fn::Function`: Decay function (default: exponential)
"""
struct DDCRPParams{T<:Real}
    α::T
    scale::T
    decay_fn::Function
end

# Default constructor with exponential decay
function DDCRPParams(α::T, scale::T) where {T<:Real}
    DDCRPParams(α, scale, (d; scale=scale) -> exp(-d * scale))
end
