# ============================================================================
# Data Simulation Utilities
# ============================================================================

using Distributions, Random

"""
    simulate_m(a, b, tables)

Simulate cluster means from InverseGamma(a, b) prior.
Returns a vector of means, one per table.
"""
function simulate_m(a, b, tables)
    return [rand(InverseGamma(a, b)) for _ in tables]
end

"""
    simulate_λ(customer_assignments, tables, m, r)

Simulate latent rates λ from Gamma(r, m[k]/r) for each observation.
Each observation gets a rate drawn from its cluster's Gamma distribution.

# Arguments
- `customer_assignments`: Customer link vector
- `tables`: Vector of tables (from table_vector)
- `m`: Vector of cluster means (one per table)
- `r`: Dispersion parameter
"""
function simulate_λ(customer_assignments, tables, m, r)
    n = length(customer_assignments)
    λ = zeros(n)

    for (k, table) in enumerate(tables)
        # Gamma(r, m/r) has mean m and variance m²/r
        for i in table
            λ[i] = rand(Gamma(r, m[k] / r))
        end
    end

    return λ
end

"""
    simulate_negbin_data(n, cluster_means, r; α=0.1, scale=1.0, x=nothing)

Simulate complete negative binomial data with DDCRP clustering.

# Arguments
- `n`: Number of observations
- `cluster_means`: True cluster means (recycled if needed)
- `r`: Dispersion parameter
- `α`: DDCRP concentration parameter (default 0.1)
- `scale`: DDCRP distance scale (default 1.0)
- `x`: Optional pre-specified 1D covariate vector of length `n`. If `nothing`
  (default), positions are drawn uniformly from [0, 1]. Pass a structured
  vector to control which observations occupy which spatial region.

# Returns
- `y`: Observed counts
- `λ`: True latent rates
- `c`: Customer assignments
- `tables`: Table structure
- `m`: Cluster means used
- `x`: Covariate (used to construct distance)
- `D`: Distance matrix
"""
function simulate_negbin_data(n::Int, cluster_means::Vector{Float64}, r::Float64;
                               α::Float64=0.1, scale::Float64=1.0,
                               x::Union{Nothing, Vector{Float64}}=nothing)
    # Generate 1D covariate and distance matrix
    if isnothing(x)
        x = rand(n)
    else
        length(x) == n || throw(ArgumentError("x must have length n=$n, got $(length(x))"))
    end
    D = construct_distance_matrix(x)

    # Simulate DDCRP clustering
    c = simulate_ddcrp(D; α=α, scale=scale)
    tables = table_vector(c)
    n_clusters = length(tables)

    # Assign cluster means (recycle if fewer provided than clusters)
    m = [cluster_means[mod1(k, length(cluster_means))] for k in 1:n_clusters]

    # Simulate latent rates
    λ = simulate_λ(c, tables, m, r)

    # Simulate observed counts
    y = rand.(Poisson.(λ))

    return (y=y, λ=λ, c=c, tables=tables, m=m, x=x, D=D)
end

"""
    simulate_poisson_data(n, cluster_rates; α=0.1, scale=1.0)

Simulate Poisson data with DDCRP clustering.
"""
function simulate_poisson_data(n::Int, cluster_rates::Vector{Float64};
                                α::Float64=0.1, scale::Float64=1.0)
    x = rand(n)
    D = construct_distance_matrix(x)

    c = simulate_ddcrp(D; α=α, scale=scale)
    tables = table_vector(c)
    n_clusters = length(tables)

    # Assign cluster rates
    λ_clusters = [cluster_rates[mod1(k, length(cluster_rates))] for k in 1:n_clusters]

    # Assign rate to each observation
    λ = zeros(n)
    for (k, table) in enumerate(tables)
        for i in table
            λ[i] = λ_clusters[k]
        end
    end

    y = rand.(Poisson.(λ))

    return (y=y, λ=λ, c=c, tables=tables, x=x, D=D)
end

"""
    simulate_binomial_data(n, N, cluster_probs; α=0.1, scale=1.0)

Simulate Binomial data with DDCRP clustering.

# Arguments
- `n`: Number of observations
- `N`: Number of trials (scalar or vector)
- `cluster_probs`: True cluster success probabilities
"""
function simulate_binomial_data(n::Int, N::Union{Int, Vector{Int}}, cluster_probs::Vector{Float64};
                                 α::Float64=0.1, scale::Float64=1.0)
    x = rand(n)
    D = construct_distance_matrix(x)

    c = simulate_ddcrp(D; α=α, scale=scale)
    tables = table_vector(c)
    n_clusters = length(tables)

    # Assign cluster probabilities
    p_clusters = [cluster_probs[mod1(k, length(cluster_probs))] for k in 1:n_clusters]

    # Assign probability to each observation
    p = zeros(n)
    for (k, table) in enumerate(tables)
        for i in table
            p[i] = p_clusters[k]
        end
    end

    # Simulate counts
    N_vec = N isa Int ? fill(N, n) : N
    y = [rand(Binomial(N_vec[i], p[i])) for i in 1:n]

    return (y=y, p=p, c=c, tables=tables, N=N_vec, x=x, D=D)
end

"""
    simulate_skewnormal_data(n, cluster_ξ, cluster_ω, cluster_α; α=0.1, scale=1.0)

Simulate Skew Normal data with DDCRP clustering.

# Arguments
- `n`: Number of observations
- `cluster_ξ`: True cluster location parameters
- `cluster_ω`: True cluster scale parameters
- `cluster_α`: True cluster shape parameters

# Returns
Named tuple with:
- `y`: Observed continuous values
- `ξ`: Location per observation
- `ω`: Scale per observation
- `α_shape`: Shape per observation (named α_shape to avoid conflict with DDCRP α)
- `c`: Customer assignments
- `tables`: Table structure
- `x`: Covariate (used to construct distance)
- `D`: Distance matrix
"""
function simulate_skewnormal_data(n::Int,
                                   cluster_ξ::Vector{Float64},
                                   cluster_ω::Vector{Float64},
                                   cluster_α::Vector{Float64};
                                   α::Float64=0.1, scale::Float64=1.0)
    x = rand(n)
    D = construct_distance_matrix(x)

    c = simulate_ddcrp(D; α=α, scale=scale)
    tables = table_vector(c)
    n_clusters = length(tables)

    # Assign cluster parameters (recycle if fewer provided than clusters)
    ξ_clusters = [cluster_ξ[mod1(k, length(cluster_ξ))] for k in 1:n_clusters]
    ω_clusters = [cluster_ω[mod1(k, length(cluster_ω))] for k in 1:n_clusters]
    α_clusters = [cluster_α[mod1(k, length(cluster_α))] for k in 1:n_clusters]

    # Assign parameters to each observation
    ξ = zeros(n)
    ω = zeros(n)
    α_shape = zeros(n)

    for (k, table) in enumerate(tables)
        for i in table
            ξ[i] = ξ_clusters[k]
            ω[i] = ω_clusters[k]
            α_shape[i] = α_clusters[k]
        end
    end

    # Simulate observations using stochastic representation
    # y = ξ + ω × (δ × |z| + √(1-δ²) × ε)
    y = zeros(n)
    for i in 1:n
        δ = α_shape[i] / sqrt(1 + α_shape[i]^2)
        τ = 1 - δ^2
        z = abs(randn())  # Half-normal
        ε = randn()       # Standard normal
        y[i] = ξ[i] + ω[i] * (δ * z + sqrt(τ) * ε)
    end

    return (y=y, ξ=ξ, ω=ω, α_shape=α_shape, c=c, tables=tables, x=x, D=D)
end

"""
    simulate_gamma_data(n, cluster_shapes, cluster_rates; α=0.1, scale=1.0)

Simulate Gamma data with DDCRP clustering.

# Arguments
- `n`: Number of observations
- `cluster_shapes`: True cluster shape parameters (α_k)
- `cluster_rates`: True cluster rate parameters (β_k)
- `α`: DDCRP concentration parameter
- `scale`: DDCRP distance scale

# Returns
Named tuple with:
- `y`: Observed positive continuous values
- `α_shape`: Shape per observation (named α_shape to avoid conflict with DDCRP α)
- `β`: Rate per observation
- `c`: Customer assignments
- `tables`: Table structure
- `x`: Covariate (used to construct distance)
- `D`: Distance matrix
"""
function simulate_gamma_data(n::Int,
                              cluster_shapes::Vector{Float64},
                              cluster_rates::Vector{Float64};
                              α::Float64=0.1, scale::Float64=1.0)
    x = rand(n)
    D = construct_distance_matrix(x)

    c = simulate_ddcrp(D; α=α, scale=scale)
    tables = table_vector(c)
    n_clusters = length(tables)

    # Assign cluster parameters (recycle if fewer provided than clusters)
    α_clusters = [cluster_shapes[mod1(k, length(cluster_shapes))] for k in 1:n_clusters]
    β_clusters = [cluster_rates[mod1(k, length(cluster_rates))] for k in 1:n_clusters]

    # Assign parameters to each observation
    α_shape = zeros(n)
    β = zeros(n)

    for (k, table) in enumerate(tables)
        for i in table
            α_shape[i] = α_clusters[k]
            β[i] = β_clusters[k]
        end
    end

    # Simulate observations from Gamma(α, β) where β is rate parameter
    # Julia's Gamma uses scale=1/β, so Gamma(α, 1/β)
    y = [rand(Gamma(α_shape[i], 1/β[i])) for i in 1:n]

    return (y=y, α_shape=α_shape, β=β, c=c, tables=tables, x=x, D=D)
end
