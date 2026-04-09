# ============================================================================
# test_prediction.jl
#
# Validates the posterior predictive for PoissonPopulationRatesMarg.
#
# Setup:
#   - 90 observations in 3 spatial clusters (30 each)
#   - Covariates: 1D Gaussian blobs at 0, 5, 10
#   - Distance matrix: Absolute difference on covariates
#   - True rates: ρ = [0.5, 2.0, 5.0] (well-separated)
#   - Population: P_i ~ Uniform(500, 5000)
#   - y_i ~ Poisson(P_i * ρ_{z_i})
#   - Mask: 10 observations per cluster (30 total) -> posterior predictive
# ============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DDCRP, Random, Distributions, Statistics
using Printf
using Plots, StatsPlots
using StatsBase

Random.seed!(42)

# ============================================================================
# Simulate data
# ============================================================================

n_per_cluster = 30
n_clusters    = 3
n             = n_per_cluster * n_clusters

# Cluster centres in 1D covariate space
centres = [0.0, 5.0, 10.0]
ρ_true  = [0.5, 2.0, 5.0]

# True cluster labels (1-indexed)
z_true = repeat(1:n_clusters, inner=n_per_cluster)

# Generate 1D covariates
σ_cov = 1.0
X = zeros(n)
for (k, cx) in enumerate(centres)
    idx = z_true .== k
    X[idx] = cx .+ σ_cov .* randn(n_per_cluster)
end

# Absolute distance matrix
D = [abs(X[i] - X[j]) for i in 1:n, j in 1:n]

# Population offsets
P = rand(500:5000, n)

# Observations
y_true = [rand(Poisson(P[i] * ρ_true[z_true[i]])) for i in 1:n]


scatter(X, y_true ./ P)

# ============================================================================
# Mask observations
# ============================================================================

# Mask 10 per cluster (indices 21:30 in each cluster block)
mask_per_cluster = 10
masked_idx = Int[]
for k in 1:n_clusters
    block_start = (k - 1) * n_per_cluster + 1
    append!(masked_idx, block_start + n_per_cluster - mask_per_cluster : block_start + n_per_cluster - 1)
end

y_obs = Vector{Union{Int, Missing}}(y_true)
for i in masked_idx
    y_obs[i] = missing
end

println("Total observations: $n")
println("Masked:             $(length(masked_idx))")
println("Observed:           $(n - length(masked_idx))")

# ============================================================================
# Run MCMC
# ============================================================================

model       = PoissonPopulationRatesMarg()
ddcrp_params = DDCRPParams(1.0, 2.0)
priors      = PoissonPopulationRatesMargPriors(1.0, 0.5)
opts        = MCMCOptions(
    n_samples=5000, 
    verbose=true,
    infer_params= Dict(
        :α_ddcrp => true, 
        :s_ddcrp => false
    ),
)

println("\nRunning MCMC...")
samples, _ = mcmc(
    model, y_true, P, D, ddcrp_params, priors, ConjugateProposal(); opts=opts
)

plot(calculate_n_clusters(samples.c))

# Reconstruct data object as mcmc() sees it (with missing_mask)
missing_mask = BitVector(ismissing.(y_obs))
y_filled = Int.(coalesce.(y_obs, 0))
data = CountDataWithPopulation(y_filled, P, D, missing_mask)

# ============================================================================
# Posterior predictive
# ============================================================================

println("\nGenerating posterior predictive samples...")
pred, pred_idx = posterior_predictive(model, samples, data, priors)

# Burn-in: discard first 20%
burn = div(opts.n_samples, 5)
pred_post = pred[(burn+1):end, :]



# ============================================================================
# Evaluation
# ============================================================================

println("\n=== Posterior Predictive Summary ===\n")
println(@sprintf("%-6s  %-6s  %-10s  %-10s  %-12s  %-8s  %-6s",
                 "Idx", "Clust", "True y", "Pred mean", "Pred median", "95% CI", "Cover"))
println(repeat("-", 72))

covered_vec = Bool[]
for (k, j) in enumerate(pred_idx)
    draws = pred_post[:, k]
    lo, hi = quantile(draws, [0.025, 0.975])
    covered = lo <= y_true[j] <= hi
    push!(covered_vec, covered)
    clust = z_true[j]
    println(@sprintf("%-6d  %-6d  %-10d  %-10.1f  %-12.0f  [%5.0f,%5.0f]  %s",
                     j, clust, y_true[j], mean(draws), median(draws), lo, hi,
                     covered ? "yes" : "NO"))
end

n_covered = sum(covered_vec)
coverage = 100.0 * n_covered / length(pred_idx)
println(repeat("-", 72))
@printf("\n95%% CI coverage: %.1f%% (%d / %d)\n", coverage, n_covered, length(pred_idx))

# Median absolute error
mae_pred  = median([abs(median(pred_post[:, k]) - y_true[j]) for (k, j) in enumerate(pred_idx)])
# Naive baseline: draw from prior only
mae_prior = let
    prior_draws = [rand(Poisson(P[j] * rand(Gamma(priors.ρ_a, 1.0/priors.ρ_b)))) for j in pred_idx]
    median(abs.(prior_draws .- [y_true[j] for j in pred_idx]))
end
@printf("Median |error| (posterior predictive): %.1f\n", mae_pred)
@printf("Median |error| (prior predictive):     %.1f\n", mae_prior)

# Cluster breakdown
println("\nCoverage by cluster:")
for k in 1:n_clusters
    mask_k = [z_true[j] == k for j in pred_idx]
    cov_k  = sum([(quantile(pred_post[:, i], 0.025) <= y_true[pred_idx[i]] <= quantile(pred_post[:, i], 0.975))
                  for i in findall(mask_k)])
    n_k    = sum(mask_k)
    @printf("  Cluster %d (ρ=%.1f): %d / %d covered (%.0f%%)\n",
            k, ρ_true[k], cov_k, n_k, 100.0 * cov_k / n_k)
end

# ============================================================================
# Plots
# ============================================================================

burn = div(opts.n_samples, 5)
c_post = samples.c[(burn+1):end, :]

# --- Plot 1: Posterior number of clusters ---
n_clusters_trace = calculate_n_clusters(c_post)
k_vals  = sort(unique(n_clusters_trace))
k_probs = [mean(n_clusters_trace .== k) for k in k_vals]

p1 = bar(k_vals, k_probs;
    xlabel="Number of clusters K",
    ylabel="Posterior probability",
    title="Posterior distribution of K\n(true K = $n_clusters)",
    legend=false,
    color=:steelblue,
    xticks=k_vals)
vline!(p1, [n_clusters]; color=:red, linestyle=:dash, linewidth=2, label="True K")

# --- Plot 2: Posterior predictive distributions by cluster ---
# Gather draws per masked obs, grouped by true cluster
cluster_labels  = [z_true[j] for j in pred_idx]
cluster_colours = [:steelblue, :darkorange, :seagreen]

p2 = plot(; xlabel="Observation index (masked)",
            ylabel="Count",
            title="Posterior predictive vs true values",
            legend=:topleft)

for k in 1:n_clusters
    k_idx = findall(cluster_labels .== k)
    # 95% CI bars
    for i in k_idx
        draws = pred_post[:, i]
        lo, hi = quantile(draws, [0.025, 0.975])
        obs_i  = pred_idx[i]
        plot!(p2, [obs_i, obs_i], [lo, hi];
              color=cluster_colours[k], linewidth=2, label=nothing)
    end
    # Posterior mean dots
    means_k = [mean(pred_post[:, i]) for i in k_idx]
    scatter!(p2, pred_idx[k_idx], means_k;
             color=cluster_colours[k], markershape=:circle,
             markersize=5, label="Cluster $k (ρ=$(ρ_true[k]))")
end
# True values
scatter!(p2, pred_idx, [y_true[j] for j in pred_idx];
         color=:black, markershape=:xcross, markersize=6, label="True y")

# --- Plot 3: Predictive draw distributions as violin per cluster ---
p3 = plot(; xlabel="True cluster", ylabel="Predicted count",
            title="Predictive draw distributions by cluster")
for k in 1:n_clusters
    k_idx  = findall(cluster_labels .== k)
    draws_k = vec(pred_post[:, k_idx])
    violin!(p3, fill(k, length(draws_k)), draws_k;
            color=cluster_colours[k], alpha=0.6, label=nothing)
    scatter!(p3, [k], [mean(y_true[pred_idx[k_idx]])];
             color=:black, markershape=:diamond, markersize=8, label=nothing)
end
xticks!(p3, 1:n_clusters, ["Cluster $k\n(ρ=$(ρ_true[k]))" for k in 1:n_clusters])

# --- Combine and save ---
fig = plot(p1, p2, p3; layout=(1, 3), size=(1600, 500), margin=8Plots.mm)
savefig(fig, joinpath(@__DIR__, "prediction_plots.png"))
println("\nPlots saved to scripts/prediction_plots.png")
display(fig)
