# ============================================================================
# DDCRP Hyperparameter Inference
# Gibbs sampling for α (self-link probability) and s (distance scale)
# using data augmentation (Appendix B of the paper)
# ============================================================================

using Distributions, Random

"""
    compute_R(s, D)

Compute R_i = Σ_{j≠i} exp(-s · d_{ij}) for all i.
These are the unnormalized total link weights from observation i to all others.
"""
function compute_R(s::Real, D::AbstractMatrix)
    n = size(D, 1)
    R = zeros(n)
    @inbounds for i in 1:n, j in 1:n
        if i != j
            R[i] += exp(-s * D[i, j])
        end
    end
    return R
end

"""
    count_self_links(c)

Count the number of self-links in the customer assignment vector c
(i.e., number of i where c[i] == i).
"""
count_self_links(c) = sum(c[i] == i for i in eachindex(c))

"""
    sample_V!(V, α, R)

Sample auxiliary variables V_i ~ Exponential(α + R_i) in-place.
These are used for data-augmented Gibbs sampling of α.

Julia's Exponential(θ) uses scale parameterisation: E[V] = θ, so
V ~ Exponential(1/(α + R_i)) gives rate = α + R_i.
"""
function sample_V!(V::AbstractVector, α::Real, R::AbstractVector)
    @inbounds for i in eachindex(V)
        V[i] = rand(Exponential(1.0 / (α + R[i])))
    end
end

"""
    update_α_ddcrp(n_self, V, ddcrp_params)

Exact Gibbs update for the DDCRP self-link parameter α.

Uses data augmentation: given auxiliary variables V_i ~ Exp(α + R_i),
the conditional posterior is conjugate:

    α | V, c, s ~ Gamma(a_α + n_self, 1 / (b_α + Σ_i V_i))

where (a_α, b_α) are the Gamma shape and rate prior parameters.

# Arguments
- `n_self`: Number of self-links in current assignment c
- `V`: Auxiliary variable vector (length n), already sampled
- `ddcrp_params`: DDCRPParams with prior fields α_a, α_b set

# Returns
- New sampled value of α
"""
function update_α_ddcrp(
    n_self::Int,
    V::AbstractVector,
    ddcrp_params::DDCRPParams
)
    a_α = ddcrp_params.α_a
    b_α = ddcrp_params.α_b
    α_new = rand(Gamma(a_α + n_self, 1.0 / (b_α + sum(V))))
    return α_new
end

"""
    update_s_ddcrp_augmented(s, α, V, R_current, c, D, ddcrp_params, prop_sd)

MH update for the DDCRP distance scale s using the data-augmented likelihood.

Uses a log-normal random walk proposal: s' = s · exp(ε), ε ~ N(0, σ²).

The log acceptance ratio (including Jacobian) is:
    log r = a_s · log(s'/s) − (b_s + D_sum) · (s' − s) − Σ_i V_i · [R_i(s') − R_i(s)]

where D_sum = Σ_{i: c_i ≠ i} d_{i,c_i} and R_i(s) = Σ_{j≠i} exp(-s·d_{ij}).

# Arguments
- `s`: Current scale value
- `V`: Auxiliary variables (already sampled), used in acceptance ratio
- `R_current`: Pre-computed R_i values for current s
- `c`: Current customer assignment vector
- `D`: Distance matrix
- `ddcrp_params`: DDCRPParams with prior fields s_a, s_b set
- `prop_sd`: Log-normal proposal standard deviation

# Returns
- New (accepted or rejected) value of s
"""
function update_s_ddcrp_augmented(
    s::Real,
    V::AbstractVector,
    R_current::AbstractVector,
    c::AbstractVector{Int},
    D::AbstractMatrix,
    ddcrp_params::DDCRPParams,
    prop_sd::Real
)
    a_s = ddcrp_params.s_a
    b_s = ddcrp_params.s_b
    n = length(c)

    # Propose s' via log-normal random walk
    s_prop = exp(log(s) + randn() * prop_sd)

    # Sum of distances for non-self links (numerator contribution)
    D_sum = 0.0
    @inbounds for i in 1:n
        if c[i] != i
            D_sum += D[i, c[i]]
        end
    end

    # R_i for proposed s
    R_prop = compute_R(s_prop, D)

    # Log acceptance ratio
    log_accept = a_s * log(s_prop / s) -
                 (b_s + D_sum) * (s_prop - s) -
                 sum(V[i] * (R_prop[i] - R_current[i]) for i in 1:n)

    if log(rand()) < log_accept
        return s_prop
    else
        return s
    end
end

"""
    update_s_ddcrp(s, α, c, D, ddcrp_params, prop_sd)

MH update for the DDCRP distance scale s using the non-augmented likelihood.
Used when α is not being inferred (no auxiliary variables available).

The log acceptance ratio is:
    log r = a_s · log(s'/s) − (b_s + D_sum) · (s' − s) − Σ_i [log Z_i(s') − log Z_i(s)]

where Z_i(s) = α + R_i(s) is the normalising constant for observation i.

# Arguments
- `s`: Current scale value
- `α`: Current (fixed) α value
- `c`: Current customer assignment vector
- `D`: Distance matrix
- `ddcrp_params`: DDCRPParams with prior fields s_a, s_b set
- `prop_sd`: Log-normal proposal standard deviation

# Returns
- New (accepted or rejected) value of s
"""
function update_s_ddcrp(
    s::Real,
    α::Real,
    c::AbstractVector{Int},
    D::AbstractMatrix,
    ddcrp_params::DDCRPParams,
    prop_sd::Real
)
    a_s = ddcrp_params.s_a
    b_s = ddcrp_params.s_b
    n = length(c)

    # Propose s' via log-normal random walk
    s_prop = exp(log(s) + randn() * prop_sd)

    # Sum of distances for non-self links
    D_sum = 0.0
    @inbounds for i in 1:n
        if c[i] != i
            D_sum += D[i, c[i]]
        end
    end

    # Compute normalising constants Z_i = α + R_i for current and proposed s
    R_current = compute_R(s, D)
    R_prop = compute_R(s_prop, D)

    log_accept = a_s * log(s_prop / s) -
                 (b_s + D_sum) * (s_prop - s) -
                 sum(log((α + R_prop[i]) / (α + R_current[i])) for i in 1:n)

    if log(rand()) < log_accept
        return s_prop
    else
        return s
    end
end
