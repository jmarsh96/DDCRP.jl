# ============================================================================
# Walk Free Foundation Global Slavery Index - ddCRP Analysis
# ============================================================================
#
# Applies Negative Binomial models with population offsets to the GSI data.
# Compares:
#   1. NBPopulationRatesMarg  – Gibbs (marginalised γ_k)
#   2. NBPopulationRates      – RJMCMC + PriorProposal + NoUpdate
#   3. NBPopulationRates      – RJMCMC + PriorProposal + WeightedMean
#
# Verifies that all configurations target the same posterior by comparing
# K distributions and pairwise co-clustering matrices.
# ============================================================================

using Distributed
using DDCRP
using CSV, DataFrames, Distances
using Statistics, StatsBase, LinearAlgebra
using Plots, StatsPlots
using Random
using Printf
using XLSX
using JLD2

Random.seed!(2025)

# ============================================================================
# 0. Output directories
# ============================================================================

mkpath("results/walkfree")
mkpath("results/walkfree/figures")
mkpath("results/walkfree/chains")

# ============================================================================
# 1. Load and preprocess data
# ============================================================================

println("Loading GSI data...")
data_path = joinpath("data", "2023-Global-Slavery-Index-Data.xlsx")
df = DataFrame(XLSX.readtable(data_path, "GSI 2023 summary data"; first_row = 3))
rename!(df, Dict(
    "Estimated number of people in modern slavery" => :est_num_ms,
    "Population" => :pop,
    "Governance issues" => :governance_issues,
    "Lack of basic needs" => :lack_basic_needs,
    "Inequality" => :inequality,
    "Disenfranchised groups" => :disenfranchised_groups,
    "Effects of conflict" => :effects_of_conflict
))

# also remove any zero pops
filter!(x -> x.pop > 0, df)

# Drop rows with missing values in the required columns
covariate_cols = [:governance_issues, :lack_basic_needs, :inequality,
                  :disenfranchised_groups, :effects_of_conflict]
required_cols  = [:est_num_ms, :pop, covariate_cols...]

df_clean = dropmissing(df, required_cols)
println("  Countries after dropping missing: $(nrow(df_clean)) / $(nrow(df))")

y = Int.(df_clean.est_num_ms)   # count of people in modern slavery
P = Int.(df_clean.pop)                  # population exposure

# Standardise covariates to zero-mean, unit-variance before computing distances
X = Float64.(Matrix(df_clean[:, covariate_cols]))
X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)

# Euclidean distance in covariate space → n×n matrix
D = pairwise(Euclidean(), X_std', dims=2)

n = length(y)
println("  n = $n observations")
println("  y range: [$(minimum(y)), $(maximum(y))]")
println("  P range: [$(minimum(P)), $(maximum(P))]")

# ============================================================================
# 2. DDCRP and prior specification
# ============================================================================

# NB population priors: γ ~ Gamma(2, rate=0.1) ⟹ E[γ]=20, r ~ Gamma(1, rate=0.1)
priors_marg   = NBPopulationRatesMargPriors(2.0, 0.1, 1.0, 0.1)
priors_unmarg = NBPopulationRatesPriors(2.0, 0.1, 1.0, 0.1)

# ============================================================================
# 2b. Cross-validation for optimal DDCRP hyperparameters (α, scale)
# ============================================================================

α_grid = 0.1:0.5:5.0
scale_grid = 0.5:0.5:10.0
n_cv       = 10_000
cv_burnin  = 2_000

cv_opts = MCMCOptions(
    n_samples         = n_cv,
    verbose           = false,
    track_diagnostics = false,
    prop_sds          = Dict(:λ => 0.3, :r => 0.5)
)

waic_grid = fill(Inf, length(α_grid), length(scale_grid))
lpml_grid = fill(-Inf, length(α_grid), length(scale_grid))
meank_grid = fill(-Inf, length(α_grid), length(scale_grid))

# Launch distributed workers for parallel CV.
# Auto-detects SLURM: if SLURM_JOB_ID is set, uses SlurmClusterManager to
# launch workers across the allocated nodes; otherwise falls back to local addprocs.
if nprocs() == 1
    exeflags = ["--project=$(Base.active_project())"]
    if haskey(ENV, "SLURM_JOB_ID")
        using SlurmClusterManager
        n_slurm_tasks = parse(Int, get(ENV, "SLURM_NTASKS", "1"))
        println("SLURM detected: SLURM_NTASKS=$n_slurm_tasks — launching $(n_slurm_tasks - 1) workers via SlurmManager")
        addprocs(SlurmManager(); exeflags=exeflags)
    else
        n_workers = 8
        println("No SLURM detected: adding $n_workers local workers")
        addprocs(n_workers; exeflags=exeflags)
    end
    @everywhere using DDCRP
    @everywhere using Statistics
    @everywhere using JLD2
end

n_fits = length(α_grid) * length(scale_grid)
println("\nRunning DDCRP parameter cross-validation ($n_fits fits on $(nworkers()) workers)...")

param_grid = [
    (ia, is, α, sc) for (ia, α) in enumerate(α_grid) for (is, sc) in enumerate(scale_grid)
]

cv_results = pmap(param_grid) do (ia, is, α, sc)
    cv_params  = DDCRPParams(α, sc)
    cv_samples = mcmc(
        NBPopulationRatesMarg(),
        y, P, D,
        cv_params,
        priors_marg,
        ConjugateProposal();
        opts = cv_opts
    )
    chain_path = @sprintf "results/walkfree/chains/cv_a%.1f_s%.1f.jld2" α sc
    @save chain_path cv_samples α sc
    res  = compute_waic(y, cv_samples.λ; burnin=cv_burnin)
    lpml = compute_lpml(y, cv_samples.λ; burnin=cv_burnin)
    mean_k = mean(calculate_n_clusters(cv_samples.c[cv_burnin+1:end, :]))
    (ia=ia, is=is, α=α, scale=sc, waic=res.waic, lpml=lpml, mean_k=mean_k)
end

for r in cv_results
    waic_grid[r.ia, r.is] = r.waic
    lpml_grid[r.ia, r.is] = r.lpml
    meank_grid[r.ia, r.is] = r.mean_k
    @printf "  α=%-5.1f  scale=%-6.1f  WAIC=%10.2f  LPML=%10.2f  mean_k=%6.2f\n" r.α r.scale r.waic r.lpml r.mean_k
end

best_idx  = argmin(waic_grid)
α_opt     = α_grid[best_idx[1]]
scale_opt = scale_grid[best_idx[2]]
println("\nOptimal α=$α_opt, scale=$scale_opt  (WAIC=$(round(waic_grid[best_idx], digits=2)))")

# Save CV grid
cv_df = DataFrame(
    α     = repeat(α_grid, inner=length(scale_grid)),
    scale = repeat(scale_grid, outer=length(α_grid)),
    waic  = vec(waic_grid),
    lpml  = vec(lpml_grid),
    mean_k = vec(meank_grid)
)
CSV.write("results/walkfree/cv_grid.csv", cv_df)

# CV heatmap plots
p_cv_waic = heatmap(scale_grid, α_grid, waic_grid,
    xlabel="scale", ylabel="α", title="WAIC surface (lower = better)",
    color=cgrad(:viridis, rev=true), colorbar_title="WAIC")
savefig(p_cv_waic, "results/walkfree/figures/cv_waic_surface.png")

p_cv_lpml = heatmap(scale_grid, α_grid, lpml_grid,
    xlabel="scale", ylabel="α", title="LPML surface (higher = better)",
    color=:viridis, colorbar_title="LPML")
savefig(p_cv_lpml, "results/walkfree/figures/cv_lpml_surface.png")

ddcrp_params = DDCRPParams(α_opt, scale_opt)

n_samples  = 160_000
n_burnin   = 10_000

base_opts = MCMCOptions(
    n_samples         = n_samples,
    verbose           = true,
    track_diagnostics = true,
    prop_sds          = Dict(:λ => 0.3, :r => 0.5)
)

# ============================================================================
# 3. Run 1 – Marginalised Gibbs sampler
# ============================================================================

println("\n[Run 1] NBPopulationRatesMarg – Gibbs (ConjugateProposal)")
samples_marg, diag_marg = mcmc(
    NBPopulationRatesMarg(),
    y, P, D,
    ddcrp_params,
    priors_marg,
    ConjugateProposal();
    opts = base_opts
)

println("  Total time: $(round(diag_marg.total_time, digits=1)) s")
@save "results/walkfree/chains/samples_marg.jld2" samples_marg diag_marg ddcrp_params priors_marg
println("  Chain saved to results/walkfree/chains/samples_marg.jld2")

plot(samples_marg.logpost)
plot(calculate_n_clusters(samples_marg.c))


# ============================================================================
# 4. Run 2 – Non-marginalised RJMCMC, PriorProposal + NoUpdate
# ============================================================================

println("\n[Run 2] NBPopulationRates – RJMCMC + PriorProposal + NoUpdate")
samples_rj, diag_rj = mcmc(
    NBPopulationRates(),
    y, P, D,
    ddcrp_params,
    priors_unmarg,
    PriorProposal();
    fixed_dim_proposal = NoUpdate(),
    opts = base_opts
)

ar_rj = acceptance_rates(diag_rj)
println("  Acceptance rates — birth: $(round(ar_rj.birth, digits=3)),  death: $(round(ar_rj.death, digits=3)),  fixed: $(round(ar_rj.fixed, digits=3))")
println("  Total time: $(round(diag_rj.total_time, digits=1)) s")
@save "results/walkfree/chains/samples_rj.jld2" samples_rj diag_rj ddcrp_params priors_unmarg
println("  Chain saved to results/walkfree/chains/samples_rj.jld2")

plot(samples_rj.logpost[200:end])
plot(calculate_n_clusters(samples_rj.c))


# ============================================================================
# 6. Post-processing helpers
# ============================================================================

function postprocess(samples, label, n_burnin)
    idx    = (n_burnin + 1):size(samples.c, 1)
    c_post = samples.c[idx, :]
    r_post = samples.r[idx]
    k_post = calculate_n_clusters(c_post)

    # ESS
    ess_k = effective_sample_size(Float64.(k_post))
    ess_r = effective_sample_size(r_post)

    println("\n--- $label (post burn-in = $(length(idx)) samples) ---")
    println("  K:  mean=$(round(mean(k_post), digits=2))  median=$(median(k_post))  mode=$(argmax(countmap(k_post)))")
    println("  r:  mean=$(round(mean(r_post), digits=3))  95%CI=[$(round(quantile(r_post,0.025),digits=3)), $(round(quantile(r_post,0.975),digits=3))]")
    println("  ESS(K)=$(round(ess_k, digits=1))  ESS(r)=$(round(ess_r, digits=1))")

    sim = compute_similarity_matrix(c_post)

    return (c=c_post, r=r_post, k=k_post, sim=sim, ess_k=ess_k, ess_r=ess_r,
            time=nothing, label=label)
end

res_marg = postprocess(samples_marg, "Marg-Gibbs",    n_burnin)
res_rj  = postprocess(samples_rj,  "RJMCMC",  n_burnin)

results = [res_marg, res_rj]

# ============================================================================
# 7. K distribution comparison
# ============================================================================

all_k = sort(unique(vcat(res_marg.k, res_rj.k)))
println("\n=== K distribution comparison ===")
@printf "%-5s  %-14s  %-14s\n" "K" "Marg-Gibbs" "RJMCMC"
println("-" ^ 55)
for k in all_k
    p1 = round(mean(res_marg.k .== k), digits=4)
    p2 = round(mean(res_rj.k  .== k), digits=4)
    @printf "%-5d  %-14.4f  %-14.4f\n" k p1 p2
end

# ============================================================================
# 8. Plots
# ============================================================================

colors = [:steelblue, :darkorange, :forestgreen]
labels = ["Marg-Gibbs", "RJMCMC", "RJMCMC-WtMean"]

# 8a. K distribution bar chart (overlay)
p_k = plot(title="Posterior K distribution (Walk Free)", xlabel="K", ylabel="Probability")
for (res, col, lbl) in zip(results, colors, labels)
    cm  = countmap(res.k)
    ks  = sort(collect(keys(cm)))
    ps  = [cm[k] / length(res.k) for k in ks]
    plot!(p_k, ks, ps, label=lbl, color=col, markershape=:circle, linewidth=2)
end
savefig(p_k, "results/walkfree/figures/k_distribution.png")

# 8b. r trace / density
p_r_trace = plot(title="r traces (post burn-in)", xlabel="Iteration", ylabel="r")
for (res, col, lbl) in zip(results, colors, labels)
    plot!(p_r_trace, res.r, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_r_trace, "results/walkfree/figures/r_trace.png")

p_r_dens = plot(title="Posterior density of r", xlabel="r", ylabel="Density")
for (res, col, lbl) in zip(results, colors, labels)
    density!(p_r_dens, res.r, label=lbl, color=col, linewidth=2)
end
savefig(p_r_dens, "results/walkfree/figures/r_density.png")

# 8c. K trace
p_k_trace = plot(title="Number of clusters K (post burn-in)", xlabel="Iteration", ylabel="K")
for (res, col, lbl) in zip(results, colors, labels)
    plot!(p_k_trace, res.k, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_k_trace, "results/walkfree/figures/k_trace.png")

# 8d. Co-clustering heatmaps
for (res, lbl) in zip(results, labels)
    p_sim = heatmap(res.sim, title="Co-clustering: $lbl", xlabel="Country", ylabel="Country",
                    color=:viridis, colorbar_title="Pr(same cluster)", aspect_ratio=:equal)
    savefig(p_sim, "results/walkfree/figures/coclustering_$(replace(lbl, r"[^A-Za-z0-9]+" => "_")).png")
end

# 8e. Logpost traces
p_lp = plot(title="Log-posterior traces", xlabel="Iteration", ylabel="Log-posterior")
post_samples = [samples_marg, samples_rj]
for (s, col, lbl) in zip(post_samples, colors, labels)
    plot!(p_lp, s.logpost[(n_burnin+1):end], label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_lp, "results/walkfree/figures/logpost_trace.png")

println("\nAll figures saved to results/walkfree/figures/")

# ============================================================================
# 9. Summary metrics CSV
# ============================================================================

ar_names  = ["marg_gibbs", "rjmcmc_no_update"]
ar_list   = [nothing, ar_rj]
time_list = [diag_marg.total_time, diag_rj.total_time]

rows = DataFrame(
    run            = ar_names,
    mean_K         = [round(mean(r.k), digits=3) for r in results],
    median_K       = [median(r.k) for r in results],
    mode_K         = [argmax(countmap(r.k)) for r in results],
    mean_r         = [round(mean(r.r), digits=4) for r in results],
    r_ci_lo        = [round(quantile(r.r, 0.025), digits=4) for r in results],
    r_ci_hi        = [round(quantile(r.r, 0.975), digits=4) for r in results],
    ess_K          = [round(r.ess_k, digits=1) for r in results],
    ess_r          = [round(r.ess_r, digits=1) for r in results],
    total_time_s   = [round(t, digits=1) for t in time_list],
    birth_acc_rate = [isnothing(ar) ? NaN : round(ar.birth, digits=4) for ar in ar_list],
    death_acc_rate = [isnothing(ar) ? NaN : round(ar.death, digits=4) for ar in ar_list],
    fixed_acc_rate = [isnothing(ar) ? NaN : round(ar.fixed, digits=4) for ar in ar_list],
)

CSV.write("results/walkfree/summary_metrics.csv", rows)
println("Summary metrics saved to results/walkfree/summary_metrics.csv")
println("CV grid saved to results/walkfree/cv_grid.csv")
println("All figures in results/walkfree/figures/")

# ============================================================================
# 10. Cluster visualisation (PCA-based, singletons removed)
# ============================================================================

# -- PCA via SVD (LinearAlgebra already loaded; X_std is n × p, zero-mean) --
F_svd    = svd(X_std)
pc1      = F_svd.U[:, 1] .* F_svd.S[1]
pc2      = F_svd.U[:, 2] .* F_svd.S[2]
var_expl = 100 .* F_svd.S .^ 2 ./ sum(F_svd.S .^ 2)
pct1     = round(var_expl[1], digits=1)
pct2     = round(var_expl[2], digits=1)

# -- MAP cluster labels from marginalised Gibbs (post burn-in) --
z_map       = point_estimate_clustering(res_marg.c; method=:MAP)
clust_sizes = countmap(z_map)
keep_mask   = [clust_sizes[z_map[i]] > 1 for i in 1:n]
keep_idx    = findall(keep_mask)
n_keep      = length(keep_idx)

unique_cl = sort(unique(z_map[keep_idx]))
K_ns      = length(unique_cl)
remap     = Dict(c => k for (k, c) in enumerate(unique_cl))
z_ns      = [remap[z_map[i]] for i in keep_idx]

println("\nCluster plots: $(n - n_keep) singletons removed, " *
        "$n_keep countries, $K_ns non-trivial clusters")

# Local indices within keep_idx belonging to cluster k
cl_idx_fn(k) = [j for (j, z) in enumerate(z_ns) if z == k]

# ── Plot 1: PC1 vs count (log scale), coloured by MAP cluster ───────────────
p_pc1_count = plot(
    xlabel = "PC1 ($pct1% var explained)",
    ylabel = "People in modern slavery",
    title  = "PC1 vs Count — MAP clusters (singletons excluded)",
    legend = :outerright,
    yscale = :log10,
    size   = (900, 500)
)
for k in 1:K_ns
    ji = cl_idx_fn(k)
    scatter!(p_pc1_count,
        pc1[keep_idx[ji]], Float64.(y[keep_idx[ji]]);
        label = "Cluster $k", markersize = 5, markerstrokewidth = 0.4
    )
end
savefig(p_pc1_count, "results/walkfree/figures/cluster_pc1_vs_count.png")

# ── Plot 2: PC1 vs PC2, coloured by MAP cluster ─────────────────────────────
p_pca = plot(
    xlabel = "PC1 ($pct1% var explained)",
    ylabel = "PC2 ($pct2% var explained)",
    title  = "PCA — MAP clusters (singletons excluded)",
    legend = :outerright,
    size   = (900, 500)
)
for k in 1:K_ns
    ji = cl_idx_fn(k)
    scatter!(p_pca,
        pc1[keep_idx[ji]], pc2[keep_idx[ji]];
        label = "Cluster $k", markersize = 5, markerstrokewidth = 0.4
    )
end
savefig(p_pca, "results/walkfree/figures/cluster_pca.png")

# ── Posterior directed link probabilities P(c[i] = j) ──────────────────────
# link_prob[i, j] = fraction of posterior samples where customer i links to j
link_prob = zeros(Float64, n, n)
for s in axes(res_marg.c, 1)
    for i in 1:n
        j = res_marg.c[s, i]
        link_prob[i, j] += 1.0
    end
end
link_prob ./= size(res_marg.c, 1)

link_thresh = 0.05   # only draw arrows where P(c[i]=j) ≥ this

# ── Plot 3: PC1 vs PC2 with directed arrows for P(c[i] = j) ────────────────
p_arrows = plot(
    xlabel = "PC1 ($pct1% var explained)",
    ylabel = "PC2 ($pct2% var explained)",
    title  = "Posterior link probabilities P(cᵢ=j) — singletons excluded",
    legend = :outerright,
    size   = (900, 550)
)
# Draw arrows first so scatter points appear on top
for i in keep_idx, j in keep_idx
    i == j && continue
    p_ij = link_prob[i, j]
    p_ij < link_thresh && continue
    plot!(p_arrows,
        [pc1[i], pc1[j]], [pc2[i], pc2[j]];
        label     = "",
        color     = :gray,
        alpha     = min(p_ij * 2, 0.8),
        linewidth = 0.5 + 2.5 * p_ij,
        arrow     = true
    )
end
# Scatter points on top, coloured by MAP cluster
for k in 1:K_ns
    ji = cl_idx_fn(k)
    scatter!(p_arrows,
        pc1[keep_idx[ji]], pc2[keep_idx[ji]];
        label = "Cluster $k", markersize = 5, markerstrokewidth = 0.4
    )
end
savefig(p_arrows, "results/walkfree/figures/cluster_link_arrows.png")

println("Cluster visualisation plots saved to results/walkfree/figures/")
println("\nDone.")
