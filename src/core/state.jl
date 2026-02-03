# ============================================================================
# MCMC State Container Types
# ============================================================================
# Note: State types are now defined in their respective model files
# (e.g., NBGammaPoissonGlobalRMargState in nb_gamma_poisson_global_r_marg.jl)

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

