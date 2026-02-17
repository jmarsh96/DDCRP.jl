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
# ============================================================================
# Generic MixedProposal Dispatch
# ============================================================================
#
# These @generated functions iterate over the NamedTuple of per-parameter
# proposals at compile time, producing type-stable, allocation-free code.
# Each parameter dispatches to sample_birth_param / birth_param_logpdf,
# which are implemented per (model, Val{param}, proposal) in each model file.

@generated function sample_birth_params(
    model::LikelihoodModel,
    prop::MixedProposal{T},
    S_i::Vector{Int},
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors
) where {T<:NamedTuple}
    names = fieldnames(T)
    n = length(names)
    vals  = [Symbol("_val_$i")  for i in 1:n]
    logqs = [Symbol("_logq_$i") for i in 1:n]
    stmts = Expr[]
    for (i, name) in enumerate(names)
        push!(stmts, quote
            ($(vals[i]), $(logqs[i])) = sample_birth_param(
                model, Val($(QuoteNode(name))), prop.proposals.$name,
                S_i, state, data, priors
            )
        end)
    end
    params_expr = :(NamedTuple{$names}(($(vals...),)))
    logq_expr   = length(logqs) == 1 ? logqs[1] : :(+($(logqs...)))
    return quote
        $(stmts...)
        return $params_expr, $logq_expr
    end
end

@generated function birth_params_logpdf(
    model::LikelihoodModel,
    prop::MixedProposal{T},
    params_old::NamedTuple,
    S_i::Vector{Int},
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors
) where {T<:NamedTuple}
    names = fieldnames(T)
    logqs = [Symbol("_logq_$i") for i in 1:length(names)]
    stmts = Expr[]
    for (i, name) in enumerate(names)
        push!(stmts, quote
            $(logqs[i]) = birth_param_logpdf(
                model, Val($(QuoteNode(name))), prop.proposals.$name,
                params_old.$name, S_i, state, data, priors
            )
        end)
    end
    total_expr = length(logqs) == 1 ? logqs[1] : :(+($(logqs...)))
    return quote
        $(stmts...)
        return $total_expr
    end
end

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
