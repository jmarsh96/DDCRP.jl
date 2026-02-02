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
    simulate_negbin_data(n, n_clusters, cluster_means, r; α=0.1, scale=1.0)

Simulate complete negative binomial data with DDCRP clustering.

# Arguments
- `n`: Number of observations
- `n_clusters`: Approximate number of clusters (via α)
- `cluster_means`: True cluster means (recycled if needed)
- `r`: Dispersion parameter

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
                               α::Float64=0.1, scale::Float64=1.0)
    # Generate 1D covariate and distance matrix
    x = rand(n)
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
