# ============================================================================
# UK National Referral Mechanism (NRM) — ddCRP County Clustering Analysis
# ============================================================================
#
# Models NRM referral counts per UK county (Table_12 of the quarterly statistics
# ODS file) as Negative Binomial counts with population exposure offsets.
# Compares:
#   1. NBPopulationRatesMarg  – Gibbs (marginalised γ_k)
#   2. NBPopulationRates      – RJMCMC + PriorProposal + NoUpdate
#
# Covariates driving the distance kernel: % aged 15–29, employment rate,
# % non-white, and a harmonised deprivation score (IMD/SIMD/WIMD/NIMDM).
# α (concentration) and s (decay scale) are jointly inferred.
# ============================================================================

using DDCRP
using CSV, DataFrames, Distances
using Statistics, StatsBase, LinearAlgebra
using Plots, StatsPlots
using Random
using Printf
using JLD2
using Distributions
using OdsIO

Random.seed!(2025)

# ============================================================================
# Test-run mode
# Pass `--test` on the command line (or set TEST_RUN=true) to use few
# iterations for a quick smoke-test of the full pipeline.
# ============================================================================

const TEST_RUN = "--test" in ARGS || get(ENV, "TEST_RUN", "false") == "true"
if TEST_RUN
    println("*** TEST RUN MODE — reduced iteration counts ***")
end

# ============================================================================
# 0. Output directories
# ============================================================================

mkpath("results/nrm")
mkpath("results/nrm/figures")
mkpath("results/nrm/chains")
mkpath("results/nrm/tables")

# ============================================================================
# 1. Load ODS data (Table_12) via OdsIO
# ============================================================================

println("Loading NRM Table_12 data...")

ods_path = joinpath("data", "national-referral-mechanism-statistics-uk-quarter-4-2025-oct-to-dec-tables.ods")

# Table_12 has a 5-row header before the data starts; OdsIO returns all rows so we
# skip manually.  The sheet has two columns: county name and total referrals.
raw = ods_read(ods_path; sheetName="Table_12", retType="DataFrame")

# Find the header row ("UK County") and extract data rows below it
header_row = findfirst(x -> !ismissing(x) && string(x) == "UK County", raw[:, 1])
if isnothing(header_row)
    error("Could not locate 'UK County' header in Table_12")
end

data_rows = raw[(header_row + 1):end, :]

county_col   = 1   # first column: county name
referral_col = 2   # second column: total referrals

nrm_df = DataFrame(
    county   = String[],
    referrals = Int[],
)
for row in eachrow(data_rows)
    cname = row[county_col]
    rval  = row[referral_col]
    ismissing(cname) && continue
    s = strip(string(cname))
    length(s) == 0 && continue
    r = ismissing(rval) ? 0 : Int(round(Float64(rval)))
    push!(nrm_df, (county=s, referrals=r))
end

println("  Counties in ODS: $(nrow(nrm_df))")

# ============================================================================
# 2. Load covariate data and join
# ============================================================================

println("Loading covariate data...")
cov_df = CSV.read(joinpath("data", "uk_county_covariates.csv"), DataFrame)

df = innerjoin(nrm_df, cov_df, on = :county)
println("  Counties after join with covariate CSV: $(nrow(df)) / $(nrow(nrm_df))")

# ============================================================================
# 3. Preprocessing
# ============================================================================

y = Int.(df.referrals)           # NRM referral count
P = Int.(df.population)          # population exposure

covariate_cols = [:pct_age_15_29, :employment_rate, :pct_non_white, :deprivation_score]
X     = Float64.(Matrix(df[:, covariate_cols]))
X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)

# Euclidean distance in standardised covariate space → n×n matrix
D = pairwise(Euclidean(), X_std', dims=2)

let dists = [D[i,j] for i in axes(D,1) for j in (i+1):last(axes(D,2))]
    display(histogram(dists, xlabel="Distance", ylabel="Count",
                      title="Pairwise Covariate Distances (upper tri)", legend=false))
end

n = length(y)
println("  n = $n counties")
println("  y range: [$(minimum(y)), $(maximum(y))]")
println("  P range: [$(minimum(P)), $(maximum(P))]")

D_offdiag = [D[i,j] for i in 1:n, j in 1:n if i != j]
println("  D (off-diagonal) mean=$(round(mean(D_offdiag),digits=3))  median=$(round(median(D_offdiag),digits=3))  max=$(round(maximum(D_offdiag),digits=3))")
for s_test in [0.1, 0.5, 1.0, 2.0, 5.0]
    pct = round(100 * mean(exp.(-s_test .* D_offdiag) .> 0.01), digits=1)
    println("    s=$s_test: $(pct)% of pairs have decay > 0.01")
end

# ============================================================================
# 4. DDCRP and prior specification
# ============================================================================

# γ ~ Gamma(1, rate=0.1) → E[γ]=10; r ~ Gamma(1, rate=0.1)
priors_marg   = NBPopulationRatesMargPriors(1.0, 0.1, 1.0, 0.1)
priors_unmarg = NBPopulationRatesPriors(1.0, 0.1, 1.0, 0.1)

# α prior: Gamma(1, 0.01) — diffuse, E[α]=100
# s prior: Gamma(1, 0.01) — diffuse, E[s]=100
ddcrp_params = DDCRPParams(1.0, 1.0, 1.0, 0.01, 1.0, 0.01)

if TEST_RUN
    n_samples = 2_000
    n_burnin  = 500
else
    n_samples = 20_000
    n_burnin  = 5_000
end

n_burnin < n_samples || error("n_burnin ($n_burnin) must be less than n_samples ($n_samples)")

opts = MCMCOptions(
    n_samples         = n_samples,
    verbose           = true,
    track_diagnostics = true,
    infer_params      = Dict(:r => true, :c => true, :α_ddcrp => true, :s_ddcrp => true),
    prop_sds          = Dict(:r => 0.5, :s_ddcrp => 0.3)
)

colors = [:steelblue, :darkorange]
labels = ["Marg-Gibbs", "RJMCMC"]

# ============================================================================
# Helper functions
# ============================================================================

function posterior_predictive_samples(λ_post::Matrix{Float64}, P::Vector{Int})
    n_iter, n_obs = size(λ_post)
    y_pred = zeros(Int, n_iter, n_obs)
    for s in 1:n_iter
        for i in 1:n_obs
            y_pred[s, i] = rand(Poisson(P[i] * λ_post[s, i]))
        end
    end
    return y_pred
end

function ppc_scatter_panel(ypred, y, col, title_str)
    pp_mean = vec(mean(Float64.(ypred), dims=1))
    pp_lo   = [quantile(Float64.(ypred[:, i]), 0.025) for i in 1:length(y)]
    pp_hi   = [quantile(Float64.(ypred[:, i]), 0.975) for i in 1:length(y)]
    yf      = Float64.(y)

    p = plot(title=title_str, xlabel="Observed y", ylabel="PP mean",
             xscale=:log10, yscale=:log10, legend=false)
    for i in eachindex(y)
        if yf[i] > 0
            plot!(p, [yf[i], yf[i]], [max(1.0, pp_lo[i]), max(1.0, pp_hi[i])];
                  color=col, alpha=0.3, linewidth=0.8)
        end
    end
    scatter!(p, max.(1.0, yf), max.(1.0, pp_mean); color=col, markersize=3,
             markerstrokewidth=0.3, alpha=0.7)
    xy_range = [1.0, maximum(max.(1.0, yf))]
    plot!(p, xy_range, xy_range; color=:black, linestyle=:dash, linewidth=1)
    return p
end

function postprocess(samples, label, n_burnin)
    idx    = (n_burnin + 1):size(samples.c, 1)
    c_post = samples.c[idx, :]
    r_post = samples.r[idx]
    k_post = calculate_n_clusters(c_post)
    α_post = samples.α_ddcrp[idx]
    s_post = samples.s_ddcrp[idx]

    ess_k = effective_sample_size(Float64.(k_post))
    ess_r = effective_sample_size(r_post)
    ess_α = effective_sample_size(α_post)
    ess_s = effective_sample_size(s_post)

    println("\n--- $label (post burn-in = $(length(idx)) samples) ---")
    println("  K:  mean=$(round(mean(k_post), digits=2))  median=$(median(k_post))  mode=$(argmax(countmap(k_post)))")
    println("  r:  mean=$(round(mean(r_post), digits=3))  95%CI=[$(round(quantile(r_post,0.025),digits=3)), $(round(quantile(r_post,0.975),digits=3))]")
    println("  α:  mean=$(round(mean(α_post), digits=3))  95%CI=[$(round(quantile(α_post,0.025),digits=3)), $(round(quantile(α_post,0.975),digits=3))]")
    println("  s:  mean=$(round(mean(s_post), digits=3))  95%CI=[$(round(quantile(s_post,0.025),digits=3)), $(round(quantile(s_post,0.975),digits=3))]")
    println("  ESS(K)=$(round(ess_k, digits=1))  ESS(r)=$(round(ess_r, digits=1))  ESS(α)=$(round(ess_α, digits=1))  ESS(s)=$(round(ess_s, digits=1))")

    sim = compute_similarity_matrix(c_post)

    return (c=c_post, r=r_post, k=k_post, α=α_post, s=s_post, sim=sim,
            ess_k=ess_k, ess_r=ess_r, ess_α=ess_α, ess_s=ess_s, label=label)
end

# ============================================================================
# 5. Run 1 – Marginalised Gibbs sampler (α + s inferred)
# ============================================================================

println("\n[Run 1] NBPopulationRatesMarg – Gibbs (ConjugateProposal)")
samples_marg, diag_marg = mcmc(
    NBPopulationRatesMarg(),
    y, P, D,
    ddcrp_params,
    priors_marg,
    ConjugateProposal();
    opts = opts
)
println("  Total time: $(round(diag_marg.total_time, digits=1)) s")
@save "results/nrm/chains/samples_marg.jld2" samples_marg diag_marg ddcrp_params priors_marg
println("  Chain saved to results/nrm/chains/samples_marg.jld2")

# ============================================================================
# 6. Run 2 – Non-marginalised RJMCMC, PriorProposal + NoUpdate (α + s inferred)
# ============================================================================

println("\n[Run 2] NBPopulationRates – RJMCMC + PriorProposal + NoUpdate")
samples_rj, diag_rj = mcmc(
    NBPopulationRates(),
    y, P, D,
    ddcrp_params,
    priors_unmarg,
    PriorProposal();
    fixed_dim_proposal = NoUpdate(),
    opts = opts
)
ar_rj = acceptance_rates(diag_rj)
println("  Acceptance rates — birth: $(round(ar_rj.birth, digits=3)),  death: $(round(ar_rj.death, digits=3)),  fixed: $(round(ar_rj.fixed, digits=3))")
println("  Total time: $(round(diag_rj.total_time, digits=1)) s")
@save "results/nrm/chains/samples_rj.jld2" samples_rj diag_rj ddcrp_params priors_unmarg
println("  Chain saved to results/nrm/chains/samples_rj.jld2")

# ============================================================================
# 7. Post-processing
# ============================================================================

res_marg = postprocess(samples_marg, "Marg-Gibbs", n_burnin)
res_rj   = postprocess(samples_rj,   "RJMCMC",     n_burnin)

results = [res_marg, res_rj]

# ============================================================================
# 7b. Extended diagnostics — cluster membership, co-clustering, posterior params
# ============================================================================

counties = String.(df.county)

for (res, lbl) in zip(results, labels)
    println("\n=== Extended diagnostics: $lbl ===")

    z_map  = point_estimate_clustering(res.c; method=:MAP)
    csizes = countmap(z_map)
    println("  MAP cluster composition:")
    for (cl, cnt) in sort(collect(csizes), by=x -> x[2])
        members = sort(counties[findall(==(cl), z_map)])
        println("    Cluster $cl (n=$cnt): " * join(members, ", "))
    end

    k_cm     = countmap(res.k)
    k_sorted = sort(collect(k_cm), by=x -> -x[2])
    println("  K posterior (top 10):")
    for (k, cnt) in k_sorted[1:min(10, length(k_sorted))]
        @printf "    K=%-4d  p=%.4f\n" k cnt/length(res.k)
    end

    big_cl  = argmax(csizes)
    big_idx = findall(==(big_cl), z_map)
    if length(big_idx) > 1
        sub_sim = res.sim[big_idx, big_idx]
        off = [sub_sim[i,j] for i in 1:length(big_idx), j in 1:length(big_idx) if i != j]
        println("  Co-clustering within biggest cluster (n=$(length(big_idx))): mean=$(round(mean(off),digits=3))  min=$(round(minimum(off),digits=3))")
    end

    s_mean    = mean(res.s)
    half_life = log(2) / s_mean
    println("  At posterior mean s=$(round(s_mean,digits=3)): decay half-life d=$(round(half_life,digits=3))  (D max=$(round(maximum(D),digits=2)))")
end

# ============================================================================
# 8. K distribution comparison
# ============================================================================

all_k = sort(unique(vcat(res_marg.k, res_rj.k)))
println("\n=== K distribution comparison ===")
@printf "%-5s  %-14s  %-14s\n" "K" "Marg-Gibbs" "RJMCMC"
println("-" ^ 40)
for k in all_k
    p1 = round(mean(res_marg.k .== k), digits=4)
    p2 = round(mean(res_rj.k  .== k), digits=4)
    @printf "%-5d  %-14.4f  %-14.4f\n" k p1 p2
end

# ============================================================================
# 9. Plots
# ============================================================================

# 9a. K distribution
p_k = plot(title="Posterior K distribution (NRM counties)", xlabel="K", ylabel="Probability")
for (res, col, lbl) in zip(results, colors, labels)
    cm  = countmap(res.k)
    ks  = sort(collect(keys(cm)))
    ps  = [cm[k] / length(res.k) for k in ks]
    plot!(p_k, ks, ps, label=lbl, color=col, markershape=:circle, linewidth=2)
end
savefig(p_k, "results/nrm/figures/k_distribution.png")

# 9b. α trace / density
p_α_trace = plot(title="α traces (post burn-in)", xlabel="Iteration", ylabel="α")
for (res, col, lbl) in zip(results, colors, labels)
    plot!(p_α_trace, res.α, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_α_trace, "results/nrm/figures/alpha_trace.png")

p_α_dens = plot(title="Posterior density of α", xlabel="α", ylabel="Density")
for (res, col, lbl) in zip(results, colors, labels)
    density!(p_α_dens, res.α, label=lbl, color=col, linewidth=2)
end
savefig(p_α_dens, "results/nrm/figures/alpha_density.png")

# 9c. s trace / density
p_s_trace = plot(title="s traces (post burn-in)", xlabel="Iteration", ylabel="s (decay scale)")
for (res, col, lbl) in zip(results, colors, labels)
    plot!(p_s_trace, res.s, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_s_trace, "results/nrm/figures/scale_trace.png")

p_s_dens = plot(title="Posterior density of scale s", xlabel="s (decay scale)", ylabel="Density")
for (res, col, lbl) in zip(results, colors, labels)
    density!(p_s_dens, res.s, label=lbl, color=col, linewidth=2)
end
savefig(p_s_dens, "results/nrm/figures/scale_density.png")

# 9d. r trace / density
p_r_trace = plot(title="r traces (post burn-in)", xlabel="Iteration", ylabel="r")
for (res, col, lbl) in zip(results, colors, labels)
    plot!(p_r_trace, res.r, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_r_trace, "results/nrm/figures/r_trace.png")

p_r_dens = plot(title="Posterior density of r", xlabel="r", ylabel="Density")
for (res, col, lbl) in zip(results, colors, labels)
    density!(p_r_dens, res.r, label=lbl, color=col, linewidth=2)
end
savefig(p_r_dens, "results/nrm/figures/r_density.png")

# 9e. K trace
p_k_trace = plot(title="Number of clusters K (post burn-in)", xlabel="Iteration", ylabel="K")
for (res, col, lbl) in zip(results, colors, labels)
    plot!(p_k_trace, res.k, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_k_trace, "results/nrm/figures/k_trace.png")

# 9f. Co-clustering heatmaps
for (res, lbl) in zip(results, labels)
    p_sim = heatmap(res.sim, title="Co-clustering: $lbl", xlabel="County", ylabel="County",
                    color=:viridis, colorbar_title="Pr(same cluster)", aspect_ratio=:equal)
    savefig(p_sim, "results/nrm/figures/coclustering_$(replace(lbl, r"[^A-Za-z0-9]+" => "_")).png")
end

# 9g. Log-posterior traces
p_lp = plot(title="Log-posterior traces", xlabel="Iteration", ylabel="Log-posterior")
for (s, col, lbl) in zip([samples_marg, samples_rj], colors, labels)
    plot!(p_lp, s.logpost[(n_burnin+1):end], label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_lp, "results/nrm/figures/logpost_trace.png")

# 9h. 4-panel DDCRP parameter posteriors
p_α_mg = plot(title="α — Marg-Gibbs", xlabel="α", ylabel="Density")
density!(p_α_mg, res_marg.α; label="", color=colors[1], linewidth=2)

p_α_rj = plot(title="α — RJMCMC", xlabel="α", ylabel="Density")
density!(p_α_rj, res_rj.α; label="", color=colors[2], linewidth=2)

p_s_mg = plot(title="s — Marg-Gibbs", xlabel="s (decay scale)", ylabel="Density")
density!(p_s_mg, res_marg.s; label="", color=colors[1], linewidth=2)

p_s_rj = plot(title="s — RJMCMC", xlabel="s (decay scale)", ylabel="Density")
density!(p_s_rj, res_rj.s; label="", color=colors[2], linewidth=2)

p_ddcrp_panel = plot(p_α_mg, p_α_rj, p_s_mg, p_s_rj, layout=(2, 2), size=(900, 600))
savefig(p_ddcrp_panel, "results/nrm/figures/ddcrp_params_posteriors.png")

# 9i. PCA cluster scatter plots (singletons removed)
F_svd    = svd(X_std)
pc1      = F_svd.U[:, 1] .* F_svd.S[1]
pc2      = F_svd.U[:, 2] .* F_svd.S[2]
var_expl = 100 .* F_svd.S .^ 2 ./ sum(F_svd.S .^ 2)
pct1     = round(var_expl[1], digits=1)
pct2     = round(var_expl[2], digits=1)

y_lo = max(1.0, 10.0 ^ floor(log10(max(1, minimum(y)))))
y_hi = 10.0 ^ ceil(log10(maximum(y)))

for (res, tag, lbl) in zip(results, ["marg", "rj"], labels)
    z_map    = point_estimate_clustering(res.c; method=:MAP)
    csizes   = countmap(z_map)
    keep_idx = findall(i -> csizes[z_map[i]] > 1, 1:n)
    n_keep   = length(keep_idx)
    K_ns     = length(unique(z_map[keep_idx]))

    println("\n  $lbl: $(n - n_keep) singletons removed, $n_keep counties, $K_ns non-trivial clusters")

    if K_ns == 0
        println("  (no non-trivial clusters — skipping PC plots)")
        continue
    end

    unique_cl = sort(unique(z_map[keep_idx]))
    remap     = Dict(c => k for (k, c) in enumerate(unique_cl))
    z_ns      = [remap[z_map[i]] for i in keep_idx]
    cl_fn(k)  = [j for (j, z) in enumerate(z_ns) if z == k]

    println("  Cluster composition ($lbl):")
    for k in 1:K_ns
        members = sort(counties[keep_idx[cl_fn(k)]])
        println("    Cluster $k (n=$(length(members))): $(join(members, ", "))")
    end

    # PC1 vs PC2
    p_pca = plot(
        xlabel = "PC1 ($pct1% var explained)",
        ylabel = "PC2 ($pct2% var explained)",
        title  = "PCA — MAP clusters ($lbl)",
        legend = :outerright,
        size   = (900, 500)
    )
    for k in 1:K_ns
        ji = cl_fn(k)
        scatter!(p_pca, pc1[keep_idx[ji]], pc2[keep_idx[ji]];
                 label="Cluster $k", markersize=5, markerstrokewidth=0.4)
    end
    savefig(p_pca, "results/nrm/figures/cluster_pca_$(tag).png")

    # PC1 vs referral count (log scale)
    p_pc1 = plot(
        xlabel = "PC1 ($pct1% var explained)",
        ylabel = "NRM referrals",
        title  = "PC1 vs Count — MAP clusters ($lbl)",
        legend = :outerright,
        yscale = :log10,
        ylims  = (y_lo, y_hi),
        size   = (900, 500)
    )
    for k in 1:K_ns
        ji = cl_fn(k)
        scatter!(p_pc1, pc1[keep_idx[ji]], max.(1.0, Float64.(y[keep_idx[ji]]));
                 label="Cluster $k", markersize=5, markerstrokewidth=0.4)
    end
    savefig(p_pc1, "results/nrm/figures/cluster_pc1_vs_count_$(tag).png")
end

println("\nAll figures saved to results/nrm/figures/")

# ============================================================================
# 10. Summary metrics CSV
# ============================================================================

ar_list   = [nothing, ar_rj]
time_list = [diag_marg.total_time, diag_rj.total_time]

rows = DataFrame(
    run            = ["marg_gibbs", "rjmcmc_no_update"],
    mean_K         = [round(mean(r.k), digits=3) for r in results],
    median_K       = [median(r.k) for r in results],
    mode_K         = [argmax(countmap(r.k)) for r in results],
    mean_r         = [round(mean(r.r), digits=4) for r in results],
    r_ci_lo        = [round(quantile(r.r, 0.025), digits=4) for r in results],
    r_ci_hi        = [round(quantile(r.r, 0.975), digits=4) for r in results],
    mean_α         = [round(mean(r.α), digits=4) for r in results],
    α_ci_lo        = [round(quantile(r.α, 0.025), digits=4) for r in results],
    α_ci_hi        = [round(quantile(r.α, 0.975), digits=4) for r in results],
    mean_s         = [round(mean(r.s), digits=4) for r in results],
    s_ci_lo        = [round(quantile(r.s, 0.025), digits=4) for r in results],
    s_ci_hi        = [round(quantile(r.s, 0.975), digits=4) for r in results],
    ess_K          = [round(r.ess_k, digits=1) for r in results],
    ess_r          = [round(r.ess_r, digits=1) for r in results],
    ess_α          = [round(r.ess_α, digits=1) for r in results],
    ess_s          = [round(r.ess_s, digits=1) for r in results],
    total_time_s   = [round(t, digits=1) for t in time_list],
    birth_acc_rate = [isnothing(ar) ? NaN : round(ar.birth, digits=4) for ar in ar_list],
    death_acc_rate = [isnothing(ar) ? NaN : round(ar.death, digits=4) for ar in ar_list],
    fixed_acc_rate = [isnothing(ar) ? NaN : round(ar.fixed, digits=4) for ar in ar_list],
)

CSV.write("results/nrm/tables/summary_metrics.csv", rows)
println("Summary metrics saved to results/nrm/tables/summary_metrics.csv")

# ============================================================================
# 11. Posterior predictive distribution
# ============================================================================

println("\n[Section 11] Computing posterior predictive distributions...")

λ_post_marg = samples_marg.λ[(n_burnin+1):end, :]
λ_post_rj   = samples_rj.λ[(n_burnin+1):end, :]

ypred_marg = posterior_predictive_samples(λ_post_marg, P)
ypred_rj   = posterior_predictive_samples(λ_post_rj,   P)

# Density of log10(y_pred+1) vs log10(y_obs+1)
p_ppc_dens = plot(
    title  = "Posterior predictive check — marginal density (log scale)",
    xlabel = "log₁₀(NRM referrals + 1)",
    ylabel = "Density",
)
for (ypred, col, lbl) in zip([ypred_marg, ypred_rj], colors, labels)
    density!(p_ppc_dens, vec(log10.(Float64.(ypred) .+ 1));
             label="PP — $lbl", color=col, linewidth=1.5, alpha=0.7)
end
density!(p_ppc_dens, log10.(Float64.(y) .+ 1);
         label="Observed", color=:black, linewidth=2, linestyle=:dash)
savefig(p_ppc_dens, "results/nrm/figures/ppc_density.png")

# PP mean vs observed scatter
p_ppc_marg    = ppc_scatter_panel(ypred_marg, y, colors[1], "PP mean vs observed — Marg-Gibbs")
p_ppc_rj      = ppc_scatter_panel(ypred_rj,   y, colors[2], "PP mean vs observed — RJMCMC")
p_ppc_scatter = plot(p_ppc_marg, p_ppc_rj, layout=(1, 2), size=(1000, 450))
savefig(p_ppc_scatter, "results/nrm/figures/ppc_mean_vs_observed.png")

println("  PPC figures saved to results/nrm/figures/")

# ============================================================================
# 12. County-level summary table
# ============================================================================

println("\n[Section 12] Writing county-level summary table...")

function county_table(ypred, λ_post, label_str, counties, y, P)
    n_obs = length(y)

    rate_mean  = [mean(λ_post[:, i])            for i in 1:n_obs]
    rate_ci_lo = [quantile(λ_post[:, i], 0.025) for i in 1:n_obs]
    rate_ci_hi = [quantile(λ_post[:, i], 0.975) for i in 1:n_obs]

    yp = Float64.(ypred)
    pp_mean   = [mean(yp[:, i])                for i in 1:n_obs]
    pp_median = [median(yp[:, i])              for i in 1:n_obs]
    pp_ci_lo  = [quantile(yp[:, i], 0.025)    for i in 1:n_obs]
    pp_ci_hi  = [quantile(yp[:, i], 0.975)    for i in 1:n_obs]

    df_out = DataFrame(
        county     = counties,
        population = P,
        y_observed = y,
        rate_mean  = round.(rate_mean,  digits=6),
        rate_ci_lo = round.(rate_ci_lo, digits=6),
        rate_ci_hi = round.(rate_ci_hi, digits=6),
        pp_mean    = round.(pp_mean,    digits=1),
        pp_median  = round.(pp_median,  digits=1),
        pp_ci_lo   = round.(pp_ci_lo,   digits=1),
        pp_ci_hi   = round.(pp_ci_hi,   digits=1),
    )
    sort!(df_out, :y_observed, rev=true)
    fname = "results/nrm/tables/county_summary_$(label_str).csv"
    CSV.write(fname, df_out)
    println("  Saved: $fname")
    return df_out
end

tbl_marg = county_table(ypred_marg, λ_post_marg, "marg_gibbs", counties, y, P)
tbl_rj   = county_table(ypred_rj,   λ_post_rj,   "rjmcmc",    counties, y, P)

# ============================================================================
# 13. Per-county posterior predictive density plots (4×3 grids)
# ============================================================================

println("\n[Section 13] Plotting per-county posterior predictive distributions...")

n_per_page = 12   # 4 rows × 3 columns
n_pages    = ceil(Int, n / n_per_page)

for page in 1:n_pages
    idx_start = (page - 1) * n_per_page + 1
    idx_end   = min(page * n_per_page, n)
    idx_page  = idx_start:idx_end

    panels = Any[]
    for i in idx_page
        yobs    = Float64(y[i])
        yp_marg = Float64.(ypred_marg[:, i])
        yp_rj   = Float64.(ypred_rj[:, i])

        p = plot(title=counties[i], titlefontsize=6,
                 xlabel="", ylabel="", legend=false, framestyle=:box,
                 tickfontsize=5)
        density!(p, yp_marg; color=colors[1], linewidth=1.2, alpha=0.8)
        density!(p, yp_rj;   color=colors[2], linewidth=1.2, alpha=0.8)
        vline!(p, [yobs]; color=:black, linewidth=1.5, linestyle=:dash)
        push!(panels, p)
    end

    while length(panels) < n_per_page
        push!(panels, plot(axis=false, grid=false, framestyle=:none))
    end

    p_grid = plot(panels..., layout=(4, 3), size=(1000, 1100),
                  plot_title=@sprintf("PP by county (page %d/%d) — Marg-Gibbs (blue), RJMCMC (orange), observed (dashed)", page, n_pages),
                  plot_titlefontsize=8)
    fname = @sprintf("results/nrm/figures/ppc_counties_%03d.png", page)
    savefig(p_grid, fname)
    println("  Saved: $fname  (counties $(idx_start)–$(idx_end) of $n)")
end

println("  Per-county PPC plots complete.")
println("\nDone.")

# ============================================================================
# 14. Fixed hyperparameter sensitivity study
# ============================================================================
#
# Runs NBPopulationRatesMarg (Gibbs) over a grid of (α, s) values with
# α and s held fixed (not inferred).  Results saved to results/nrm_fixed/.
# ============================================================================

println("\n" * "=" ^ 76)
println("SECTION 14: Fixed hyperparameter sensitivity study")
println("=" ^ 76)

mkpath("results/nrm_fixed/figures")
mkpath("results/nrm_fixed/chains")
mkpath("results/nrm_fixed/tables")

# Grid of (α, s) combinations to evaluate
fixed_grid = [
    (α=1.0,  s=0.5),
    (α=1.0,  s=1.0),
    (α=1.0,  s=2.0),
    (α=5.0,  s=0.5),
    (α=5.0,  s=1.0),
    (α=5.0,  s=2.0),
    (α=10.0, s=0.5),
    (α=10.0, s=1.0),
    (α=10.0, s=2.0),
]

opts_fixed = MCMCOptions(
    n_samples         = n_samples,
    verbose           = false,
    track_diagnostics = true,
    infer_params      = Dict(:r => true, :c => true, :α_ddcrp => false, :s_ddcrp => false),
    prop_sds          = Dict(:r => 0.5)
)

# Colour palette cycling for the grid (tab10-style)
_palette    = [:steelblue, :darkorange, :green, :red, :purple, :brown, :pink, :gray, :olive]
grid_colors = _palette[1:length(fixed_grid)]

# Containers for cross-run comparison plots
p_k_all  = plot(title="Posterior K — fixed hyperparams", xlabel="K", ylabel="Probability")
p_r_all  = plot(title="Posterior r — fixed hyperparams", xlabel="r", ylabel="Density")
p_lp_all = plot(title="Log-posterior traces — fixed hyperparams", xlabel="Iteration", ylabel="Log-posterior")

fixed_summary_rows = []

for (idx, cfg) in enumerate(fixed_grid)
    α_val = cfg.α
    s_val = cfg.s
    tag   = @sprintf("alpha%.1f_s%.1f", α_val, s_val)
    lbl   = "α=$(α_val), s=$(s_val)"
    col   = grid_colors[idx]

    println("\n[$idx/$(length(fixed_grid))] Running NBPopulationRatesMarg — $lbl")

    ddcrp_fixed = DDCRPParams(α_val, s_val)   # 2-arg: fixed α and s, no priors
    samples_f, diag_f = mcmc(
        NBPopulationRatesMarg(),
        y, P, D,
        ddcrp_fixed,
        NBPopulationRatesMargPriors(1.0, 0.1, 1.0, 0.1),
        ConjugateProposal();
        opts = opts_fixed
    )
    println("  Total time: $(round(diag_f.total_time, digits=1)) s")

    # Save chain
    @save "results/nrm_fixed/chains/samples_$(tag).jld2" samples_f diag_f α_val s_val
    println("  Chain saved to results/nrm_fixed/chains/samples_$(tag).jld2")

    # --- Post-processing (post burn-in) ---
    idx_post = (n_burnin + 1):size(samples_f.c, 1)
    c_post   = samples_f.c[idx_post, :]
    r_post   = samples_f.r[idx_post]
    k_post   = calculate_n_clusters(c_post)

    ess_k = effective_sample_size(Float64.(k_post))
    ess_r = effective_sample_size(r_post)

    println("  K:  mean=$(round(mean(k_post), digits=2))  median=$(median(k_post))  mode=$(argmax(countmap(k_post)))")
    println("  r:  mean=$(round(mean(r_post), digits=3))  95%CI=[$(round(quantile(r_post,0.025),digits=3)), $(round(quantile(r_post,0.975),digits=3))]")
    println("  ESS(K)=$(round(ess_k, digits=1))  ESS(r)=$(round(ess_r, digits=1))")

    sim   = compute_similarity_matrix(c_post)
    z_map = point_estimate_clustering(c_post; method=:MAP)

    # MAP cluster membership
    csizes   = countmap(z_map)
    k_cm     = countmap(k_post)
    k_sorted = sort(collect(k_cm), by=x -> -x[2])
    println("  K posterior (top 5):")
    for (k, cnt) in k_sorted[1:min(5, length(k_sorted))]
        @printf "    K=%-4d  p=%.4f\n" k cnt/length(k_post)
    end
    println("  MAP cluster composition:")
    for (cl, cnt) in sort(collect(csizes), by=x -> x[2])
        members = sort(counties[findall(==(cl), z_map)])
        println("    Cluster $cl (n=$cnt): " * join(members, ", "))
    end

    # --- Per-run figures ---
    mkpath("results/nrm_fixed/figures/$(tag)")

    # K trace
    p_kt = plot(k_post; title="K trace — $lbl", xlabel="Iteration", ylabel="K",
                color=col, alpha=0.7, linewidth=0.8, legend=false)
    savefig(p_kt, "results/nrm_fixed/figures/$(tag)/k_trace.png")

    # K distribution
    cm_k = countmap(k_post)
    ks_v = sort(collect(keys(cm_k)))
    ps_v = [cm_k[k] / length(k_post) for k in ks_v]
    p_kd = plot(ks_v, ps_v; title="K distribution — $lbl", xlabel="K", ylabel="Probability",
                color=col, markershape=:circle, linewidth=2, legend=false)
    savefig(p_kd, "results/nrm_fixed/figures/$(tag)/k_distribution.png")

    # r trace + density
    p_rt = plot(r_post; title="r trace — $lbl", xlabel="Iteration", ylabel="r",
                color=col, alpha=0.7, linewidth=0.8, legend=false)
    savefig(p_rt, "results/nrm_fixed/figures/$(tag)/r_trace.png")

    p_rd = plot(title="r density — $lbl", xlabel="r", ylabel="Density", legend=false)
    density!(p_rd, r_post; color=col, linewidth=2)
    savefig(p_rd, "results/nrm_fixed/figures/$(tag)/r_density.png")

    # Log-posterior trace
    p_lpt = plot(samples_f.logpost[idx_post]; title="Log-posterior — $lbl",
                 xlabel="Iteration", ylabel="Log-posterior",
                 color=col, alpha=0.7, linewidth=0.8, legend=false)
    savefig(p_lpt, "results/nrm_fixed/figures/$(tag)/logpost_trace.png")

    # Co-clustering heatmap
    p_sim = heatmap(sim; title="Co-clustering: $lbl", xlabel="County", ylabel="County",
                    color=:viridis, colorbar_title="Pr(same cluster)", aspect_ratio=:equal)
    savefig(p_sim, "results/nrm_fixed/figures/$(tag)/coclustering.png")

    # PCA cluster scatter (singletons removed)
    keep_idx = findall(i -> csizes[z_map[i]] > 1, 1:n)
    if length(unique(z_map[keep_idx])) > 0
        unique_cl = sort(unique(z_map[keep_idx]))
        remap_f   = Dict(c => k for (k, c) in enumerate(unique_cl))
        z_ns_f    = [remap_f[z_map[i]] for i in keep_idx]
        K_ns_f    = length(unique_cl)
        cl_fn_f(k) = [j for (j, z) in enumerate(z_ns_f) if z == k]

        p_pca_f = plot(xlabel="PC1 ($pct1% var explained)", ylabel="PC2 ($pct2% var explained)",
                       title="PCA — MAP clusters ($lbl)", legend=:outerright, size=(900, 500))
        for k in 1:K_ns_f
            ji = cl_fn_f(k)
            scatter!(p_pca_f, pc1[keep_idx[ji]], pc2[keep_idx[ji]];
                     label="Cluster $k", markersize=5, markerstrokewidth=0.4)
        end
        savefig(p_pca_f, "results/nrm_fixed/figures/$(tag)/cluster_pca.png")
    end

    # PPC
    λ_post_f = samples_f.λ[idx_post, :]
    ypred_f  = posterior_predictive_samples(λ_post_f, P)

    p_ppc_f = ppc_scatter_panel(ypred_f, y, col, "PP mean vs observed — $lbl")
    savefig(p_ppc_f, "results/nrm_fixed/figures/$(tag)/ppc_mean_vs_observed.png")

    p_ppc_dens_f = plot(title="PPC density — $lbl", xlabel="log₁₀(referrals+1)", ylabel="Density")
    density!(p_ppc_dens_f, vec(log10.(Float64.(ypred_f) .+ 1)); label="PP", color=col, linewidth=1.5)
    density!(p_ppc_dens_f, log10.(Float64.(y) .+ 1); label="Observed", color=:black, linewidth=2, linestyle=:dash)
    savefig(p_ppc_dens_f, "results/nrm_fixed/figures/$(tag)/ppc_density.png")

    # County-level summary CSV
    county_table(ypred_f, λ_post_f, tag, counties, y, P)

    # --- Add to cross-run comparison plots ---
    plot!(p_k_all,  ks_v, ps_v;          label=lbl, color=col, markershape=:circle, linewidth=1.5)
    density!(p_r_all, r_post;            label=lbl, color=col, linewidth=1.5, alpha=0.8)
    plot!(p_lp_all, samples_f.logpost[idx_post]; label=lbl, color=col, alpha=0.6, linewidth=0.8)

    push!(fixed_summary_rows, (
        run       = tag,
        alpha     = α_val,
        s         = s_val,
        mean_K    = round(mean(k_post),           digits=3),
        median_K  = median(k_post),
        mode_K    = argmax(countmap(k_post)),
        mean_r    = round(mean(r_post),           digits=4),
        r_ci_lo   = round(quantile(r_post,0.025), digits=4),
        r_ci_hi   = round(quantile(r_post,0.975), digits=4),
        ess_K     = round(ess_k,                  digits=1),
        ess_r     = round(ess_r,                  digits=1),
        total_time_s = round(diag_f.total_time,   digits=1),
    ))

    println("  Figures saved to results/nrm_fixed/figures/$(tag)/")
end

# Save cross-run comparison plots
savefig(p_k_all,  "results/nrm_fixed/figures/k_distribution_all.png")
savefig(p_r_all,  "results/nrm_fixed/figures/r_density_all.png")
savefig(p_lp_all, "results/nrm_fixed/figures/logpost_trace_all.png")

# Summary CSV across all fixed runs
fixed_summary_df = DataFrame(fixed_summary_rows)
CSV.write("results/nrm_fixed/tables/summary_metrics.csv", fixed_summary_df)
println("\nFixed hyperparameter study complete.")
println("  Summary: results/nrm_fixed/tables/summary_metrics.csv")
println("  Figures: results/nrm_fixed/figures/")
println("\nAll done.")
