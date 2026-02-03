# ============================================================================
# Analysis Utilities
# Posterior analysis, similarity matrices, clustering metrics
# ============================================================================

using Statistics, Clustering

"""
    calculate_n_clusters(c_samples::Matrix{Int})

Calculate number of clusters for each MCMC sample.
"""
function calculate_n_clusters(c_samples::Matrix{Int})
    return [length(table_vector(c_samples[i, :])) for i in axes(c_samples, 1)]
end

# Overload for vector (single sample)
function calculate_n_clusters(c::Vector{Int})
    return length(table_vector(c))
end

"""
    posterior_num_cluster_distribution(c_samples)

Print the posterior distribution of number of clusters.
"""
function posterior_num_cluster_distribution(c_samples)
    cluster_counts = calculate_n_clusters(c_samples)
    counts = countmap(cluster_counts)
    total = sum(values(counts))
    probs = Dict(k => v / total for (k, v) in counts)

    println("Posterior distribution of number of clusters:")
    for k in sort(collect(keys(probs)))
        println("  K = $k: $(round(probs[k] * 100, digits=1))%")
    end

    return probs
end

"""
    compute_similarity_matrix(c_samples::Matrix{Int})

Compute posterior co-clustering probability matrix.
Entry (i,j) is the proportion of samples where i and j are in the same cluster.
"""
function compute_similarity_matrix(c_samples::Matrix{Int})
    n_samples, n = size(c_samples)
    sim_mat = zeros(Float64, n, n)

    for t in 1:n_samples
        z = c_to_z(c_samples[t, :], n)

        for i in 1:n
            for j in 1:n
                if z[i] == z[j]
                    sim_mat[i, j] += 1.0
                end
            end
        end
    end

    return sim_mat ./ n_samples
end

"""
    compute_ari_trace(c_samples::Matrix{Int}, c_true::Vector{Int})

Compute Adjusted Rand Index for each MCMC sample against ground truth.
"""
function compute_ari_trace(c_samples::Matrix{Int}, c_true::Vector{Int})
    n_samples, n = size(c_samples)
    ari_trace = zeros(Float64, n_samples)

    z_true = c_to_z(c_true, n)

    for t in 1:n_samples
        z_est = c_to_z(c_samples[t, :], n)
        metrics = randindex(z_true, z_est)
        ari_trace[t] = metrics[1]  # ARI is first element
    end

    return ari_trace
end

"""
    point_estimate_clustering(c_samples::Matrix{Int}; method=:MAP)

Compute a point estimate of the clustering from posterior samples.

# Methods
- `:MAP`: Most frequent clustering configuration
- `:median_K`: Sample with number of clusters closest to median
- `:posterior_mean`: Threshold similarity matrix (requires further clustering)
"""
function point_estimate_clustering(c_samples::Matrix{Int}; method::Symbol=:MAP)
    n_samples, n = size(c_samples)

    if method == :MAP
        # Convert each sample to canonical form for comparison
        z_samples = [c_to_z(c_samples[i, :], n) for i in 1:n_samples]

        # Find most frequent
        z_counts = countmap(z_samples)
        best_z = argmax(z_counts)
        return best_z

    elseif method == :median_K
        n_clusters = calculate_n_clusters(c_samples)
        median_K = median(n_clusters)

        # Find sample closest to median
        idx = argmin(abs.(n_clusters .- median_K))
        return c_to_z(c_samples[idx, :], n)

    else
        error("Unknown method: $method")
    end
end

"""
    posterior_summary(samples::AbstractMCMCSamples; burnin::Int=0)

Compute summary statistics for posterior samples.
"""
function posterior_summary(samples::AbstractMCMCSamples; burnin::Int=0)
    start_idx = burnin + 1
    c_post = samples.c[start_idx:end, :]

    n_clusters = calculate_n_clusters(c_post)

    summary = Dict{Symbol, Any}()
    summary[:n_clusters_mean] = mean(n_clusters)
    summary[:n_clusters_median] = median(n_clusters)
    summary[:n_clusters_mode] = mode(n_clusters)
    summary[:n_clusters_std] = std(n_clusters)

    if !isnothing(samples.r)
        r_post = samples.r[start_idx:end]
        summary[:r_mean] = mean(r_post)
        summary[:r_median] = median(r_post)
        summary[:r_std] = std(r_post)
        summary[:r_quantiles] = quantile(r_post, [0.025, 0.25, 0.5, 0.75, 0.975])
    end

    if !isnothing(samples.λ)
        λ_post = samples.λ[start_idx:end, :]
        summary[:λ_means] = vec(mean(λ_post, dims=1))
        summary[:λ_stds] = vec(std(λ_post, dims=1))
    end

    return summary
end

# Helper function for mode
function mode(x)
    counts = countmap(x)
    return argmax(counts)
end

# countmap helper (if not available from StatsBase)
function countmap(x)
    counts = Dict{eltype(x), Int}()
    for val in x
        counts[val] = get(counts, val, 0) + 1
    end
    return counts
end
