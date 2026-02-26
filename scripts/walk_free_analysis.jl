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

Random.seed!(2025)

# ============================================================================
# 0. Output directories
# ============================================================================

mkpath("results/walkfree")
mkpath("results/walkfree/figures")

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

α_grid     = [0.1, 0.5, 1.0, 2.0, 5.0]
scale_grid = [0.5, 1.0, 2.0, 5.0, 10.0]
n_cv       = 4_000
cv_burnin  = 1_000

cv_opts = MCMCOptions(
    n_samples         = n_cv,
    verbose           = false,
    track_diagnostics = false,
    prop_sds          = Dict(:λ => 0.3, :r => 0.5)
)

waic_grid = fill(Inf, length(α_grid), length(scale_grid))
lpml_grid = fill(-Inf, length(α_grid), length(scale_grid))

# Launch distributed workers for parallel CV
if nprocs() == 1
    n_workers = 2
    addprocs(n_workers; exeflags = ["--project=$(Base.active_project())"])
    @everywhere using DDCRP
    @everywhere using Statistics
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
    res  = compute_waic(y, cv_samples.λ; burnin=cv_burnin)
    lpml = compute_lpml(y, cv_samples.λ; burnin=cv_burnin)
    mean_k = mean(calculate_n_clusters(cv_samples.c[cv_burnin+1:end, :]))
    (ia=ia, is=is, α=α, scale=sc, waic=res.waic, lpml=lpml, mean_k=mean_k)
end

for r in cv_results
    waic_grid[r.ia, r.is] = r.waic
    lpml_grid[r.ia, r.is] = r.lpml
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
    mean_k = vec(mean_k_grid)
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
ddcrp_params = DDCRPParams(0.1, scale_opt) # testing

n_samples  = 20_000
n_burnin   = 5_000

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
res_rj  = postprocess(samples_rj,  "RJMCMC-NoUpd",  n_burnin)

results = [res_marg, res_rj]

# ============================================================================
# 7. K distribution comparison
# ============================================================================

all_k = sort(unique(vcat(res_marg.k, res_rj.k)))
println("\n=== K distribution comparison ===")
@printf "%-5s  %-14s  %-14s  %-14s\n" "K" "Marg-Gibbs" "RJMCMC-NoUpd"
println("-" * 55)
for k in all_k
    p1 = round(mean(res_marg.k .== k), digits=4)
    p2 = round(mean(res_rj.k  .== k), digits=4)
    @printf "%-5d  %-14.4f  %-14.4f  %-14.4f\n" k p1 p2
end

# ============================================================================
# 8. Plots
# ============================================================================

colors = [:steelblue, :darkorange, :forestgreen]
labels = ["Marg-Gibbs", "RJMCMC-NoUpd", "RJMCMC-WtMean"]

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
post_samples = [samples_marg, samples_rj1, samples_rj2]
for (s, col, lbl) in zip(post_samples, colors, labels)
    plot!(p_lp, s.logpost[(n_burnin+1):end], label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_lp, "results/walkfree/figures/logpost_trace.png")

println("\nAll figures saved to results/walkfree/figures/")

# ============================================================================
# 9. Summary metrics CSV
# ============================================================================

ar_names  = ["marg_gibbs", "rjmcmc_no_update", "rjmcmc_weighted_mean"]
ar_list   = [nothing, ar_rj1, ar_rj2]
time_list = [diag_marg.total_time, diag_rj1.total_time, diag_rj2.total_time]

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
println("\nDone.")
