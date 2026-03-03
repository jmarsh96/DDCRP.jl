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
- `α_a::Union{T,Nothing}`: Gamma shape prior for α (`nothing` = don't infer)
- `α_b::Union{T,Nothing}`: Gamma rate prior for α
- `s_a::Union{T,Nothing}`: Gamma shape prior for scale s (`nothing` = don't infer)
- `s_b::Union{T,Nothing}`: Gamma rate prior for scale s
"""
struct DDCRPParams{T<:Real}
    α::T
    scale::T
    decay_fn::Function
    α_a::Union{T, Nothing}
    α_b::Union{T, Nothing}
    s_a::Union{T, Nothing}
    s_b::Union{T, Nothing}
end

exp_decay(d; scale) = exp(-d * scale)

# 2-arg constructor (backward compatible): no priors, hyperparameters are fixed
function DDCRPParams(α::T, scale::T) where {T<:Real}
    DDCRPParams(α, scale, exp_decay, nothing, nothing, nothing, nothing)
end

# 4-arg constructor: Gamma prior on α only, scale is fixed
function DDCRPParams(α::T, scale::T, α_a::T, α_b::T) where {T<:Real}
    DDCRPParams{T}(α, scale, exp_decay, α_a, α_b, nothing, nothing)
end

# 6-arg constructor: with Gamma priors on α and s, exponential decay
function DDCRPParams(α::T, scale::T, α_a::T, α_b::T, s_a::T, s_b::T) where {T<:Real}
    DDCRPParams(α, scale, exp_decay, α_a, α_b, s_a, s_b)
end

# 7-arg constructor: custom decay function and priors
function DDCRPParams(α::T, scale::T, decay_fn::Function, α_a::T, α_b::T, s_a::T, s_b::T) where {T<:Real}
    DDCRPParams{T}(α, scale, decay_fn, α_a, α_b, s_a, s_b)
end
