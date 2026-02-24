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

using DDCRP
using CSV, DataFrames, Distances
using Statistics, StatsBase, LinearAlgebra
using Plots, StatsPlots
using Random
using Printf

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
df = CSV.read("GSI_cleaned.csv", DataFrame)

# Drop rows with missing values in the required columns
covariate_cols = [:governance_issues, :lack_basic_needs, :inequality,
                  :disenfranchised_groups, :effects_of_conflict]
required_cols  = [:num_people_ms_esti, :pop, covariate_cols...]

df_clean = dropmissing(df, required_cols)
println("  Countries after dropping missing: $(nrow(df_clean)) / $(nrow(df))")

y = Int.(df_clean.num_people_ms_esti)   # count of people in modern slavery
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

ddcrp_params = DDCRPParams(1.0, 1.0)   # α=1, scale=1 (relative to standardised D)

# NB population priors: γ ~ Gamma(2, rate=0.1) ⟹ E[γ]=20, r ~ Gamma(1, rate=0.1)
priors_marg   = NBPopulationRatesMargPriors(2.0, 0.1, 1.0, 0.1)
priors_unmarg = NBPopulationRatesPriors(2.0, 0.1, 1.0, 0.1)

n_samples  = 20_000
n_burnin   = 5_000

base_opts = MCMCOptions(
    n_samples        = n_samples,
    verbose          = true,
    track_diagnostics = true,
    prop_sds         = Dict(:λ => 0.3, :r => 0.5)
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

# ============================================================================
# 4. Run 2 – Non-marginalised RJMCMC, PriorProposal + NoUpdate
# ============================================================================

println("\n[Run 2] NBPopulationRates – RJMCMC + PriorProposal + NoUpdate")
samples_rj1, diag_rj1 = mcmc(
    NBPopulationRates(),
    y, P, D,
    ddcrp_params,
    priors_unmarg,
    PriorProposal();
    fixed_dim_proposal = NoUpdate(),
    opts = base_opts
)

ar_rj1 = acceptance_rates(diag_rj1)
println("  Acceptance rates — birth: $(round(ar_rj1.birth, digits=3)),  death: $(round(ar_rj1.death, digits=3)),  fixed: $(round(ar_rj1.fixed, digits=3))")
println("  Total time: $(round(diag_rj1.total_time, digits=1)) s")

# ============================================================================
# 5. Run 3 – Non-marginalised RJMCMC, PriorProposal + WeightedMean
# ============================================================================

println("\n[Run 3] NBPopulationRates – RJMCMC + PriorProposal + WeightedMean")
samples_rj2, diag_rj2 = mcmc(
    NBPopulationRates(),
    y, P, D,
    ddcrp_params,
    priors_unmarg,
    PriorProposal();
    fixed_dim_proposal = WeightedMean(),
    opts = base_opts
)

ar_rj2 = acceptance_rates(diag_rj2)
println("  Acceptance rates — birth: $(round(ar_rj2.birth, digits=3)),  death: $(round(ar_rj2.death, digits=3)),  fixed: $(round(ar_rj2.fixed, digits=3))")
println("  Total time: $(round(diag_rj2.total_time, digits=1)) s")

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
res_rj1  = postprocess(samples_rj1,  "RJMCMC-NoUpd",  n_burnin)
res_rj2  = postprocess(samples_rj2,  "RJMCMC-WtMean", n_burnin)

results = [res_marg, res_rj1, res_rj2]

# ============================================================================
# 7. K distribution comparison
# ============================================================================

all_k = sort(unique(vcat(res_marg.k, res_rj1.k, res_rj2.k)))
println("\n=== K distribution comparison ===")
@printf "%-5s  %-14s  %-14s  %-14s\n" "K" "Marg-Gibbs" "RJMCMC-NoUpd" "RJMCMC-WtMean"
println("-" * 55)
for k in all_k
    p1 = round(mean(res_marg.k .== k), digits=4)
    p2 = round(mean(res_rj1.k  .== k), digits=4)
    p3 = round(mean(res_rj2.k  .== k), digits=4)
    @printf "%-5d  %-14.4f  %-14.4f  %-14.4f\n" k p1 p2 p3
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
println("\nDone.")
