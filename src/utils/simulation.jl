# ============================================================================
# Data Simulation Utilities
# ============================================================================

using Distributions, Random

"""
    simulate_poisson_data(n, cluster_rates; α=0.1, scale=1.0, x=nothing)

Simulate Poisson data with DDCRP clustering.

# Arguments
- `x`: Optional pre-specified 1D covariate vector of length `n`. If `nothing`
  (default), positions are drawn uniformly from [0, 1].
"""
function simulate_poisson_data(n::Int, cluster_rates::Vector{Float64};
                                α::Float64=0.1, scale::Float64=1.0,
                                x::Union{Nothing, Vector{Float64}}=nothing)
    if isnothing(x)
        x = rand(n)
    else
        length(x) == n || throw(ArgumentError("x must have length n=$n, got $(length(x))"))
    end
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
    simulate_binomial_data(n, N, cluster_probs; α=0.1, scale=1.0, x=nothing)

Simulate Binomial data with DDCRP clustering.

# Arguments
- `n`: Number of observations
- `N`: Number of trials (scalar or vector)
- `cluster_probs`: True cluster success probabilities
- `x`: Optional pre-specified 1D covariate vector of length `n`. If `nothing`
  (default), positions are drawn uniformly from [0, 1].
"""
function simulate_binomial_data(n::Int, N::Union{Int, Vector{Int}}, cluster_probs::Vector{Float64};
                                 α::Float64=0.1, scale::Float64=1.0,
                                 x::Union{Nothing, Vector{Float64}}=nothing)
    if isnothing(x)
        x = rand(n)
    else
        length(x) == n || throw(ArgumentError("x must have length n=$n, got $(length(x))"))
    end
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
    simulate_gamma_data(n, cluster_shapes, cluster_rates; α=0.1, scale=1.0, x=nothing)

Simulate Gamma data with DDCRP clustering.

# Arguments
- `n`: Number of observations
- `cluster_shapes`: True cluster shape parameters (α_k)
- `cluster_rates`: True cluster rate parameters (β_k)
- `α`: DDCRP concentration parameter
- `scale`: DDCRP distance scale
- `x`: Optional pre-specified 1D covariate vector of length `n`. If `nothing`
  (default), positions are drawn uniformly from [0, 1].

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
                              α::Float64=0.1, scale::Float64=1.0,
                              x::Union{Nothing, Vector{Float64}}=nothing)
    if isnothing(x)
        x = rand(n)
    else
        length(x) == n || throw(ArgumentError("x must have length n=$n, got $(length(x))"))
    end
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
