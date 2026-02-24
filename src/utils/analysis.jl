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
    compute_vi_trace(c_samples::Matrix{Int}, c_true::Vector{Int}) -> Vector{Float64}

Variation of Information between each MCMC sample partition and the true partition.
Lower is better; 0 = perfect recovery.

VI(U, V) = H(U|V) + H(V|U) where H is conditional entropy, computed from the
contingency table of the two partitions.
"""
function compute_vi_trace(c_samples::Matrix{Int}, c_true::Vector{Int})
    n = size(c_samples, 2)
    z_true = c_to_z(c_true, n)
    bj = Dict(l => count(==(l), z_true) for l in unique(z_true))
    vi_trace = Vector{Float64}(undef, size(c_samples, 1))
    for t in axes(c_samples, 1)
        z_est = c_to_z(c_samples[t, :], n)
        ai = Dict(l => count(==(l), z_est) for l in unique(z_est))
        nij = Dict{Tuple{Int,Int}, Int}()
        for k in 1:n
            key = (z_est[k], z_true[k])
            nij[key] = get(nij, key, 0) + 1
        end
        vi = 0.0
        for ((i, j), n_ij) in nij
            vi += n_ij * (log(ai[i] / n_ij) + log(bj[j] / n_ij))
        end
        vi_trace[t] = vi / n
    end
    return vi_trace
end

"""
    compute_kl_ppd(samples, sim; burnin, n_grid) -> Float64

KL divergence KL(p_true ‖ p_ppd) from the true skew-normal mixture to the
posterior predictive density. Uses numerical grid integration over the data range.

# Arguments
- `samples`: `SkewNormalClusterSamples` from `mcmc`
- `sim`: Named tuple returned by `simulate_skewnormal_data` (fields: `y`, `c`, `ξ`, `ω`, `α_shape`)
- `burnin`: Number of initial samples to discard
- `n_grid`: Number of grid points for numerical integration (default 500)
"""
function compute_kl_ppd(samples::SkewNormalClusterSamples, sim;
                        burnin::Int=0, n_grid::Int=500)
    n = length(sim.y)
    y_min = minimum(sim.y) - 2.0
    y_max = maximum(sim.y) + 2.0
    grid = collect(range(y_min, y_max; length=n_grid))
    step = (y_max - y_min) / (n_grid - 1)

    # True density: mixture weighted by cluster size
    z_true = c_to_z(sim.c, n)
    true_labels = unique(z_true)
    p_true = zeros(n_grid)
    for lab in true_labels
        first_i = findfirst(==(lab), z_true)
        w = count(==(lab), z_true) / n
        for g in 1:n_grid
            p_true[g] += w * exp(skewnormal_logpdf(grid[g], sim.ξ[first_i],
                                                    sim.ω[first_i], sim.α_shape[first_i]))
        end
    end
    p_true .= max.(p_true, 1e-300)

    # Posterior predictive density: average over post-burnin samples
    post_range = (burnin + 1):size(samples.c, 1)
    p_est = zeros(n_grid)
    for t in post_range
        c_t = samples.c[t, :]
        z_t = c_to_z(c_t, n)
        tables_t = unique(z_t)
        K_t = length(tables_t)
        for lab in tables_t
            first_i = findfirst(==(lab), z_t)
            for g in 1:n_grid
                p_est[g] += exp(skewnormal_logpdf(grid[g], samples.ξ[t, first_i],
                                                   samples.ω[t, first_i], samples.α[t, first_i])) / K_t
            end
        end
    end
    p_est ./= length(post_range)
    p_est .= max.(p_est, 1e-300)

    # KL(p_true ‖ p_est) via rectangle rule
    kl = 0.0
    for g in 1:n_grid
        kl += p_true[g] * log(p_true[g] / p_est[g]) * step
    end
    return kl
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

"""
    compute_waic(y, λ_samples; burnin=0) -> NamedTuple

Watanabe-Akaike Information Criterion for a Poisson observation model.
Lower WAIC indicates better out-of-sample predictive fit.

The observation model is `y_i | λ_i ~ Poisson(λ_i)`, so

    lppd   = Σ_i log E_s[ p(y_i | λ_s_i) ]
    p_WAIC = Σ_i Var_s[ log p(y_i | λ_s_i) ]
    WAIC   = -2 (lppd - p_WAIC)

# Arguments
- `y::AbstractVector`: observed counts (length n)
- `λ_samples::AbstractMatrix`: posterior samples, shape (n_samples × n)
- `burnin::Int=0`: rows to discard before computing

# Returns
`NamedTuple` with fields `waic`, `lppd`, `p_waic`, `waic_i` (per-obs contributions)
"""
function compute_waic(y::AbstractVector, λ_samples::AbstractMatrix; burnin::Int=0)
    n   = length(y)
    idx = (burnin + 1):size(λ_samples, 1)
    lppd   = 0.0
    p_waic = 0.0
    waic_i = Vector{Float64}(undef, n)
    for i in 1:n
        yi   = Float64(y[i])
        ll_i = [-λ_samples[s, i] + yi * log(λ_samples[s, i]) - loggamma(yi + 1.0)
                for s in idx]
        max_ll  = maximum(ll_i)
        lppd_i  = max_ll + log(mean(exp.(ll_i .- max_ll)))
        pwaic_i = var(ll_i)
        waic_i[i] = -2.0 * (lppd_i - pwaic_i)
        lppd   += lppd_i
        p_waic += pwaic_i
    end
    w = -2.0 * (lppd - p_waic)
    return (waic=w, lppd=lppd, p_waic=p_waic, waic_i=waic_i)
end

"""
    compute_lpml(y, λ_samples; burnin=0) -> Float64

Log Pseudo-Marginal Likelihood via the Conditional Predictive Ordinate (CPO).
Higher LPML indicates better predictive fit.

    log CPO_i = -log E_s[ 1 / p(y_i | λ_s_i) ]   (harmonic-mean estimator)
    LPML      = Σ_i log CPO_i

Uses log-sum-exp for numerical stability.

# Arguments
- `y::AbstractVector`: observed counts (length n)
- `λ_samples::AbstractMatrix`: posterior samples, shape (n_samples × n)
- `burnin::Int=0`: rows to discard before computing
"""
function compute_lpml(y::AbstractVector, λ_samples::AbstractMatrix; burnin::Int=0)
    n    = length(y)
    idx  = (burnin + 1):size(λ_samples, 1)
    lpml = 0.0
    for i in 1:n
        yi   = Float64(y[i])
        ll_i = [-λ_samples[s, i] + yi * log(λ_samples[s, i]) - loggamma(yi + 1.0)
                for s in idx]
        neg_ll     = -ll_i
        max_neg_ll = maximum(neg_ll)
        lpml      += -(max_neg_ll + log(mean(exp.(neg_ll .- max_neg_ll))))
    end
    return lpml
end
