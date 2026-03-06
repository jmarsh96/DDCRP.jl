# ============================================================================
# Walk Free Foundation Global Slavery Index - ddCRP Analysis
# ============================================================================
#
# Applies Negative Binomial models with population offsets to the GSI data.
# Compares:
#   1. NBPopulationRatesMarg  – Gibbs (marginalised γ_k)
#   2. NBPopulationRates      – RJMCMC + PriorProposal + NoUpdate
#
# α (concentration) and s (decay scale) are jointly inferred.
# ============================================================================

using DDCRP
using CSV, DataFrames, Distances
using Statistics, StatsBase, LinearAlgebra
using Plots, StatsPlots
using Random
using Printf
using XLSX
using JLD2
using Distributions

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

# Remove zero populations
filter!(x -> x.pop > 0, df)

# Drop rows with missing values in the required columns
covariate_cols = [:governance_issues, :lack_basic_needs, :inequality,
                  :disenfranchised_groups, :effects_of_conflict]
required_cols  = [:est_num_ms, :pop, covariate_cols...]

df_clean = dropmissing(df, required_cols)
println("  Countries after dropping missing: $(nrow(df_clean)) / $(nrow(df))")

y = Int.(df_clean.est_num_ms)   # count of people in modern slavery
P = Int.(df_clean.pop)          # population exposure

# Standardise covariates to zero-mean, unit-variance before computing distances
X     = Float64.(Matrix(df_clean[:, covariate_cols]))
X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)

# Euclidean distance in covariate space → n×n matrix
D = pairwise(Euclidean(), X_std', dims=2)

let dists = [D[i,j] for i in axes(D,1) for j in (i+1):last(axes(D,2))]
    display(histogram(dists, xlabel="Distance", ylabel="Count",
                      title="Pairwise Covariate Distances (upper tri)", legend=false))
end

n = length(y)
println("  n = $n observations")
println("  y range: [$(minimum(y)), $(maximum(y))]")
println("  P range: [$(minimum(P)), $(maximum(P))]")

# Distance matrix diagnostics
D_offdiag = [D[i,j] for i in 1:n, j in 1:n if i != j]
println("  D (off-diagonal) mean=$(round(mean(D_offdiag),digits=3))  median=$(round(median(D_offdiag),digits=3))  max=$(round(maximum(D_offdiag),digits=3))")
for s_test in [0.1, 0.5, 1.0, 2.0, 5.0]
    pct = round(100 * mean(exp.(-s_test .* D_offdiag) .> 0.01), digits=1)
    println("    s=$s_test: $(pct)% of pairs have decay > 0.01")
end

# ============================================================================
# 2. DDCRP and prior specification
# ============================================================================

# NB population priors: γ ~ Gamma(2, rate=0.1) ⟹ E[γ]=20, r ~ Gamma(1, rate=0.1)
priors_marg   = NBPopulationRatesMargPriors(1.0, 0.1, 1.0, 0.1)
priors_unmarg = NBPopulationRatesPriors(1.0, 0.1, 1.0, 0.1)

# α prior: Gamma(1, 0.01) — diffuse, E[α]=100
# s prior: Gamma(1, 0.01) — diffuse, E[s]=100
ddcrp_params = DDCRPParams(1.0, 1.0, 1.0, 0.01, 1.0, 0.01)

if TEST_RUN
    n_samples = 2_000
    n_burnin  = 500
else
    n_samples = 30_000
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
# Pre-run
# ============================================================================

println("\n[Pre-run] NBPopulationRatesMarg – Gibbs with fixed α=2.0, s=1.0")
opts_fixed = MCMCOptions(
    n_samples         = n_samples,
    verbose           = true,
    track_diagnostics = true,
    infer_params      = Dict(:r => true, :c => true, :α_ddcrp => false, :s_ddcrp => false),
    prop_sds          = Dict(:r => 0.5, :s_ddcrp => 0.3)
)

ddcrp_params_fixed1 = DDCRPParams(1.0, 1.0, 1.0, 0.01, 1.0, 0.01)  # α fixed at 2.0, s fixed at 1.0
samples_fixed1, diag_fixed1 = mcmc(
    NBPopulationRatesMarg(),
    y, P, D,
    ddcrp_params_fixed1,
    NBPopulationRatesMargPriors(1.0, 0.1, 1.0, 0.1),
    ConjugateProposal();
    opts = opts_fixed
)

ddcrp_params_fixed2 = DDCRPParams(1.0, 10.0, 1.0, 0.01, 1.0, 0.01)  # α fixed at 2.0, s fixed at 1.0
samples_fixed2, diag_fixed2 = mcmc(
    NBPopulationRatesMarg(),
    y, P, D,
    ddcrp_params_fixed2,
    NBPopulationRatesMargPriors(1.0, 0.1, 1.0, 0.1),
    ConjugateProposal();
    opts = opts_fixed
)

# Helper functions used in PPC sections (defined here for use in pre-run block and section 10)
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
        plot!(p, [yf[i], yf[i]], [max(1.0, pp_lo[i]), max(1.0, pp_hi[i])];
              color=col, alpha=0.3, linewidth=0.8)
    end
    scatter!(p, yf, max.(1.0, pp_mean); color=col, markersize=3,
             markerstrokewidth=0.3, alpha=0.7)
    xy_range = [max(1.0, minimum(yf)), maximum(yf)]
    plot!(p, xy_range, xy_range; color=:black, linestyle=:dash, linewidth=1)
    return p
end

mkpath("results/walkfree/fixed_hyperparams")
mkpath("results/walkfree/fixed_hyperparams/figures")

# ============================================================================
# Pre-run post-processing: fixed hyperparameter comparison
# ============================================================================

fixed_labels  = ["α=1, s=1", "α=1, s=10"]
fixed_colors  = [:steelblue, :darkorange]
fixed_samples = [samples_fixed1, samples_fixed2]

function postprocess_fixed(samples, label, n_burnin)
    idx    = (n_burnin + 1):size(samples.c, 1)
    c_post = samples.c[idx, :]
    r_post = samples.r[idx]
    k_post = calculate_n_clusters(c_post)

    ess_k = effective_sample_size(Float64.(k_post))
    ess_r = effective_sample_size(r_post)

    println("\n--- $label (post burn-in = $(length(idx)) samples) ---")
    println("  K:  mean=$(round(mean(k_post), digits=2))  median=$(median(k_post))  mode=$(argmax(countmap(k_post)))")
    println("  r:  mean=$(round(mean(r_post), digits=3))  95%CI=[$(round(quantile(r_post,0.025),digits=3)), $(round(quantile(r_post,0.975),digits=3))]")
    println("  ESS(K)=$(round(ess_k, digits=1))  ESS(r)=$(round(ess_r, digits=1))")

    sim = compute_similarity_matrix(c_post)
    return (c=c_post, r=r_post, k=k_post, sim=sim, ess_k=ess_k, ess_r=ess_r, label=label)
end

res_fixed1 = postprocess_fixed(samples_fixed1, fixed_labels[1], n_burnin)
res_fixed2 = postprocess_fixed(samples_fixed2, fixed_labels[2], n_burnin)
fixed_results = [res_fixed1, res_fixed2]

# K trace
p_kt = plot(title="K trace — fixed hyperparams", xlabel="Iteration", ylabel="K")
for (res, col, lbl) in zip(fixed_results, fixed_colors, fixed_labels)
    plot!(p_kt, res.k, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_kt, "results/walkfree/fixed_hyperparams/figures/k_trace.png")

# K distribution
p_kd = plot(title="Posterior K distribution — fixed hyperparams", xlabel="K", ylabel="Probability")
for (res, col, lbl) in zip(fixed_results, fixed_colors, fixed_labels)
    cm = countmap(res.k)
    ks = sort(collect(keys(cm)))
    ps = [cm[k] / length(res.k) for k in ks]
    plot!(p_kd, ks, ps, label=lbl, color=col, markershape=:circle, linewidth=2)
end
savefig(p_kd, "results/walkfree/fixed_hyperparams/figures/k_distribution.png")

# r trace
p_rt = plot(title="r trace — fixed hyperparams", xlabel="Iteration", ylabel="r")
for (res, col, lbl) in zip(fixed_results, fixed_colors, fixed_labels)
    plot!(p_rt, res.r, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_rt, "results/walkfree/fixed_hyperparams/figures/r_trace.png")

# r density
p_rd = plot(title="Posterior density of r — fixed hyperparams", xlabel="r", ylabel="Density")
for (res, col, lbl) in zip(fixed_results, fixed_colors, fixed_labels)
    density!(p_rd, res.r, label=lbl, color=col, linewidth=2)
end
savefig(p_rd, "results/walkfree/fixed_hyperparams/figures/r_density.png")

# Log-posterior traces
p_lp_fixed = plot(title="Log-posterior traces — fixed hyperparams", xlabel="Iteration", ylabel="Log-posterior")
for (s, col, lbl) in zip(fixed_samples, fixed_colors, fixed_labels)
    plot!(p_lp_fixed, s.logpost[(n_burnin+1):end], label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_lp_fixed, "results/walkfree/fixed_hyperparams/figures/logpost_trace.png")

# Co-clustering heatmaps
for (res, lbl) in zip(fixed_results, fixed_labels)
    tag = replace(lbl, r"[^A-Za-z0-9]+" => "_")
    p_sim = heatmap(res.sim, title="Co-clustering: $lbl", xlabel="Country", ylabel="Country",
                    color=:viridis, colorbar_title="Pr(same cluster)", aspect_ratio=:equal)
    savefig(p_sim, "results/walkfree/fixed_hyperparams/figures/coclustering_$(tag).png")
end

# Cluster membership from MAP
for (res, lbl) in zip(fixed_results, fixed_labels)
    println("\n=== MAP cluster membership: $lbl ===")
    z_map  = point_estimate_clustering(res.c; method=:MAP)
    csizes = countmap(z_map)
    for (cl, cnt) in sort(collect(csizes), by=x -> x[2])
        members = sort(String.(df_clean.Country[findall(==(cl), z_map)]))
        println("  Cluster $cl (n=$cnt): " * join(members, ", "))
    end
    k_cm     = countmap(res.k)
    k_sorted = sort(collect(k_cm), by=x -> -x[2])
    println("  K posterior (top 10):")
    for (k, cnt) in k_sorted[1:min(10, length(k_sorted))]
        @printf "    K=%-4d  p=%.4f\n" k cnt/length(res.k)
    end
end

# PPC for fixed runs
λ_post_fixed1 = samples_fixed1.λ[(n_burnin+1):end, :]
λ_post_fixed2 = samples_fixed2.λ[(n_burnin+1):end, :]

ypred_fixed1 = posterior_predictive_samples(λ_post_fixed1, P)
ypred_fixed2 = posterior_predictive_samples(λ_post_fixed2, P)

p_ppc_fixed_dens = plot(
    title  = "Posterior predictive check — fixed hyperparams (log scale)",
    xlabel = "log₁₀(people in modern slavery + 1)",
    ylabel = "Density",
)
for (ypred, col, lbl) in zip([ypred_fixed1, ypred_fixed2], fixed_colors, fixed_labels)
    density!(p_ppc_fixed_dens, vec(log10.(Float64.(ypred) .+ 1));
             label="PP — $lbl", color=col, linewidth=1.5, alpha=0.7)
end
density!(p_ppc_fixed_dens, log10.(Float64.(y) .+ 1);
         label="Observed", color=:black, linewidth=2, linestyle=:dash)
savefig(p_ppc_fixed_dens, "results/walkfree/fixed_hyperparams/figures/ppc_density.png")

p_ppc_f1 = ppc_scatter_panel(ypred_fixed1, y, fixed_colors[1], "PP mean vs observed — $(fixed_labels[1])")
p_ppc_f2 = ppc_scatter_panel(ypred_fixed2, y, fixed_colors[2], "PP mean vs observed — $(fixed_labels[2])")
p_ppc_fixed_scatter = plot(p_ppc_f1, p_ppc_f2, layout=(1, 2), size=(1000, 450))
savefig(p_ppc_fixed_scatter, "results/walkfree/fixed_hyperparams/figures/ppc_mean_vs_observed.png")

# Summary metrics CSV
rows_fixed = DataFrame(
    run      = fixed_labels,
    mean_K   = [round(mean(r.k), digits=3) for r in fixed_results],
    median_K = [median(r.k) for r in fixed_results],
    mode_K   = [argmax(countmap(r.k)) for r in fixed_results],
    mean_r   = [round(mean(r.r), digits=4) for r in fixed_results],
    r_ci_lo  = [round(quantile(r.r, 0.025), digits=4) for r in fixed_results],
    r_ci_hi  = [round(quantile(r.r, 0.975), digits=4) for r in fixed_results],
    ess_K    = [round(r.ess_k, digits=1) for r in fixed_results],
    ess_r    = [round(r.ess_r, digits=1) for r in fixed_results],
)
CSV.write("results/walkfree/fixed_hyperparams/summary_metrics.csv", rows_fixed)
println("Fixed hyperparams summary saved to results/walkfree/fixed_hyperparams/summary_metrics.csv")

println("\nFixed hyperparams figures saved to results/walkfree/fixed_hyperparams/figures/")

# ============================================================================
# 3. Run 1 – Marginalised Gibbs sampler (α + s inferred)
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
@save "results/walkfree/chains/samples_marg.jld2" samples_marg diag_marg ddcrp_params priors_marg
println("  Chain saved to results/walkfree/chains/samples_marg.jld2")

# ============================================================================
# 4. Run 2 – Non-marginalised RJMCMC, PriorProposal + NoUpdate (α + s inferred)
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
@save "results/walkfree/chains/samples_rj.jld2" samples_rj diag_rj ddcrp_params priors_unmarg
println("  Chain saved to results/walkfree/chains/samples_rj.jld2")

# ============================================================================
# 5. Post-processing
# ============================================================================

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

res_marg = postprocess(samples_marg, "Marg-Gibbs", n_burnin)
res_rj   = postprocess(samples_rj,   "RJMCMC",     n_burnin)

results = [res_marg, res_rj]

# ============================================================================
# 5b. Extended diagnostics — cluster membership, co-clustering, posterior params
# ============================================================================

for (res, lbl) in zip(results, labels)
    println("\n=== Extended diagnostics: $lbl ===")

    # Cluster membership from MAP
    z_map = point_estimate_clustering(res.c; method=:MAP)
    csizes = countmap(z_map)
    println("  MAP cluster composition:")
    for (cl, cnt) in sort(collect(csizes), by=x -> x[2])
        members = sort(String.(df_clean.Country[findall(==(cl), z_map)]))
        println("    Cluster $cl (n=$cnt): " * join(members, ", "))
    end

    # K posterior distribution (top 10 most probable values)
    k_cm = countmap(res.k)
    k_sorted = sort(collect(k_cm), by=x -> -x[2])
    println("  K posterior (top 10):")
    for (k, cnt) in k_sorted[1:min(10, length(k_sorted))]
        @printf "    K=%-4d  p=%.4f\n" k cnt/length(res.k)
    end

    # Co-clustering within the biggest MAP cluster
    big_cl = argmax(csizes)
    big_idx = findall(==(big_cl), z_map)
    if length(big_idx) > 1
        sub_sim = res.sim[big_idx, big_idx]
        off = [sub_sim[i,j] for i in 1:length(big_idx), j in 1:length(big_idx) if i != j]
        println("  Co-clustering within big cluster (n=$(length(big_idx))): mean=$(round(mean(off),digits=3))  min=$(round(minimum(off),digits=3))  median=$(round(median(off),digits=3))")
    end

    # Effective decay range at posterior mean s
    s_mean = mean(res.s)
    half_life = log(2) / s_mean
    println("  At posterior mean s=$(round(s_mean,digits=3)): decay half-life d=$(round(half_life,digits=3))  (D max=$(round(maximum(D),digits=2)))")
end

# ============================================================================
# 6. K distribution comparison
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
# 7. Plots
# ============================================================================

# 7a. K distribution
p_k = plot(title="Posterior K distribution (Walk Free)", xlabel="K", ylabel="Probability")
for (res, col, lbl) in zip(results, colors, labels)
    cm  = countmap(res.k)
    ks  = sort(collect(keys(cm)))
    ps  = [cm[k] / length(res.k) for k in ks]
    plot!(p_k, ks, ps, label=lbl, color=col, markershape=:circle, linewidth=2)
end
savefig(p_k, "results/walkfree/figures/k_distribution.png")

# 7b. α trace / density
p_α_trace = plot(title="α traces (post burn-in)", xlabel="Iteration", ylabel="α")
for (res, col, lbl) in zip(results, colors, labels)
    plot!(p_α_trace, res.α, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_α_trace, "results/walkfree/figures/alpha_trace.png")

p_α_dens = plot(title="Posterior density of α", xlabel="α", ylabel="Density")
for (res, col, lbl) in zip(results, colors, labels)
    density!(p_α_dens, res.α, label=lbl, color=col, linewidth=2)
end
savefig(p_α_dens, "results/walkfree/figures/alpha_density.png")

# 7c. s trace / density
p_s_trace = plot(title="s traces (post burn-in)", xlabel="Iteration", ylabel="s (decay scale)")
for (res, col, lbl) in zip(results, colors, labels)
    plot!(p_s_trace, res.s, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_s_trace, "results/walkfree/figures/scale_trace.png")

p_s_dens = plot(title="Posterior density of scale s", xlabel="s (decay scale)", ylabel="Density")
for (res, col, lbl) in zip(results, colors, labels)
    density!(p_s_dens, res.s, label=lbl, color=col, linewidth=2)
end
savefig(p_s_dens, "results/walkfree/figures/scale_density.png")

# 7d. r trace / density
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

# 7e. K trace
p_k_trace = plot(title="Number of clusters K (post burn-in)", xlabel="Iteration", ylabel="K")
for (res, col, lbl) in zip(results, colors, labels)
    plot!(p_k_trace, res.k, label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_k_trace, "results/walkfree/figures/k_trace.png")

# 7f. Co-clustering heatmaps
for (res, lbl) in zip(results, labels)
    p_sim = heatmap(res.sim, title="Co-clustering: $lbl", xlabel="Country", ylabel="Country",
                    color=:viridis, colorbar_title="Pr(same cluster)", aspect_ratio=:equal)
    savefig(p_sim, "results/walkfree/figures/coclustering_$(replace(lbl, r"[^A-Za-z0-9]+" => "_")).png")
end

# 7g. Logpost traces
p_lp = plot(title="Log-posterior traces", xlabel="Iteration", ylabel="Log-posterior")
for (s, col, lbl) in zip([samples_marg, samples_rj], colors, labels)
    plot!(p_lp, s.logpost[(n_burnin+1):end], label=lbl, color=col, alpha=0.7, linewidth=0.8)
end
savefig(p_lp, "results/walkfree/figures/logpost_trace.png")

# 7h. 4-panel DDCRP parameter posteriors (α and s)
p_α_mg = plot(title="α — Marg-Gibbs", xlabel="α", ylabel="Density")
density!(p_α_mg, res_marg.α; label="", color=colors[1], linewidth=2)

p_α_rj = plot(title="α — RJMCMC", xlabel="α", ylabel="Density")
density!(p_α_rj, res_rj.α; label="", color=colors[2], linewidth=2)

p_s_mg = plot(title="s — Marg-Gibbs", xlabel="s (decay scale)", ylabel="Density")
density!(p_s_mg, res_marg.s; label="", color=colors[1], linewidth=2)

p_s_rj = plot(title="s — RJMCMC", xlabel="s (decay scale)", ylabel="Density")
density!(p_s_rj, res_rj.s; label="", color=colors[2], linewidth=2)

p_ddcrp_panel = plot(p_α_mg, p_α_rj, p_s_mg, p_s_rj, layout=(2, 2), size=(900, 600))
savefig(p_ddcrp_panel, "results/walkfree/figures/ddcrp_params_posteriors.png")

println("\nAll figures saved to results/walkfree/figures/")

# ============================================================================
# 8. Summary metrics CSV
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

CSV.write("results/walkfree/summary_metrics.csv", rows)
println("Summary metrics saved to results/walkfree/summary_metrics.csv")

# ============================================================================
# 9. Cluster visualisation (PCA-based, singletons removed)
# ============================================================================

# PCA via SVD (X_std is n × p, already zero-mean)
F_svd    = svd(X_std)
pc1      = F_svd.U[:, 1] .* F_svd.S[1]
pc2      = F_svd.U[:, 2] .* F_svd.S[2]
var_expl = 100 .* F_svd.S .^ 2 ./ sum(F_svd.S .^ 2)
pct1     = round(var_expl[1], digits=1)
pct2     = round(var_expl[2], digits=1)

# Posterior directed link probabilities P(c[i] = j) — from Marg-Gibbs
link_prob = zeros(Float64, n, n)
for s in axes(res_marg.c, 1)
    for i in 1:n
        j = res_marg.c[s, i]
        link_prob[i, j] += 1.0
    end
end
link_prob ./= size(res_marg.c, 1)

# MAP clusters from Marg-Gibbs (for link arrow plot)
z_map_mg  = point_estimate_clustering(res_marg.c; method=:MAP)
csizes_mg = countmap(z_map_mg)
keep_mg   = findall(i -> csizes_mg[z_map_mg[i]] > 1, 1:n)
unique_mg = sort(unique(z_map_mg[keep_mg]))
K_ns_mg   = length(unique_mg)
remap_mg  = Dict(c => k for (k, c) in enumerate(unique_mg))
z_ns_mg   = [remap_mg[z_map_mg[i]] for i in keep_mg]
cl_mg(k)  = [j for (j, z) in enumerate(z_ns_mg) if z == k]

link_thresh = 0.05
p_arrows = plot(
    xlabel = "PC1 ($pct1% var explained)",
    ylabel = "PC2 ($pct2% var explained)",
    title  = "Posterior link probabilities P(cᵢ=j) — Marg-Gibbs, singletons excluded",
    legend = :outerright,
    size   = (900, 550)
)
for i in keep_mg, j in keep_mg
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
for k in 1:K_ns_mg
    ji = cl_mg(k)
    scatter!(p_arrows,
        pc1[keep_mg[ji]], pc2[keep_mg[ji]];
        label = "Cluster $k", markersize = 5, markerstrokewidth = 0.4
    )
end
savefig(p_arrows, "results/walkfree/figures/cluster_link_arrows.png")

# PC plots and cluster membership — looped over both models
# Precompute global log10 y-axis bounds to avoid "No strict ticks found" warnings
y_lo = 10.0 ^ floor(log10(minimum(y)))
y_hi = 10.0 ^ ceil(log10(maximum(y)))

for (res, tag, lbl) in zip(results, ["marg", "rj"], labels)
    z_map    = point_estimate_clustering(res.c; method=:MAP)
    csizes   = countmap(z_map)
    keep_idx = findall(i -> csizes[z_map[i]] > 1, 1:n)
    n_keep   = length(keep_idx)
    K_ns     = length(unique(z_map[keep_idx]))

    println("\n  $lbl: $(n - n_keep) singletons removed, $n_keep countries, $K_ns non-trivial clusters")

    if K_ns == 0
        println("  (no non-trivial clusters — skipping PC plots)")
        continue
    end

    unique_cl = sort(unique(z_map[keep_idx]))
    remap     = Dict(c => k for (k, c) in enumerate(unique_cl))
    z_ns      = [remap[z_map[i]] for i in keep_idx]
    cl_fn(k)  = [j for (j, z) in enumerate(z_ns) if z == k]

    # Cluster membership
    println("  Cluster composition ($lbl):")
    for k in 1:K_ns
        members = sort(String.(df_clean.Country[keep_idx[cl_fn(k)]]))
        println("    Cluster $k (n=$(length(members))): $(join(members, ", "))")
    end

    # PC1 vs count (log scale) — explicit ylims suppresses "No strict ticks found"
    p1 = plot(
        xlabel = "PC1 ($pct1% var explained)",
        ylabel = "People in modern slavery",
        title  = "PC1 vs Count — MAP clusters ($lbl)",
        legend = :outerright,
        yscale = :log10,
        ylims  = (y_lo, y_hi),
        size   = (900, 500)
    )
    for k in 1:K_ns
        ji = cl_fn(k)
        scatter!(p1,
            pc1[keep_idx[ji]], Float64.(y[keep_idx[ji]]);
            label = "Cluster $k", markersize = 5, markerstrokewidth = 0.4
        )
    end
    savefig(p1, "results/walkfree/figures/cluster_pc1_vs_count_$(tag).png")

    # PC1 vs PC2
    p2 = plot(
        xlabel = "PC1 ($pct1% var explained)",
        ylabel = "PC2 ($pct2% var explained)",
        title  = "PCA — MAP clusters ($lbl)",
        legend = :outerright,
        size   = (900, 500)
    )
    for k in 1:K_ns
        ji = cl_fn(k)
        scatter!(p2,
            pc1[keep_idx[ji]], pc2[keep_idx[ji]];
            label = "Cluster $k", markersize = 5, markerstrokewidth = 0.4
        )
    end
    savefig(p2, "results/walkfree/figures/cluster_pca_$(tag).png")
end

println("\nCluster visualisation saved to results/walkfree/figures/")

# ============================================================================
# 10. Posterior predictive distribution
# ============================================================================


println("\n[Section 10] Computing posterior predictive distributions...")

λ_post_marg = samples_marg.λ[(n_burnin+1):end, :]
λ_post_rj   = samples_rj.λ[(n_burnin+1):end, :]

ypred_marg = posterior_predictive_samples(λ_post_marg, P)
ypred_rj   = posterior_predictive_samples(λ_post_rj,   P)

# 10a. Density of log10(y_pred+1) vs log10(y_obs+1)  [pooled across all countries & iters]
p_ppc_dens = plot(
    title  = "Posterior predictive check — marginal density (log scale)",
    xlabel = "log₁₀(people in modern slavery + 1)",
    ylabel = "Density",
)
for (ypred, col, lbl) in zip([ypred_marg, ypred_rj], colors, labels)
    density!(p_ppc_dens, vec(log10.(Float64.(ypred) .+ 1));
             label="PP — $lbl", color=col, linewidth=1.5, alpha=0.7)
end
density!(p_ppc_dens, log10.(Float64.(y) .+ 1);
         label="Observed", color=:black, linewidth=2, linestyle=:dash)
savefig(p_ppc_dens, "results/walkfree/figures/ppc_density.png")

# 10b. PP mean vs observed scatter with 95% CI (both models, side-by-side panels)
p_ppc_marg   = ppc_scatter_panel(ypred_marg, y, colors[1], "PP mean vs observed — Marg-Gibbs")
p_ppc_rj     = ppc_scatter_panel(ypred_rj,   y, colors[2], "PP mean vs observed — RJMCMC")
p_ppc_scatter = plot(p_ppc_marg, p_ppc_rj, layout=(1, 2), size=(1000, 450))
savefig(p_ppc_scatter, "results/walkfree/figures/ppc_mean_vs_observed.png")

println("  PPC figures saved to results/walkfree/figures/")

# ============================================================================
# 11. Country-level summary table
# ============================================================================

println("\n[Section 11] Writing country-level summary table...")
mkpath("results/walkfree/tables")

function country_table(ypred, λ_post, label_str, df_clean, y, P)
    n_obs = length(y)
    countries = String.(df_clean.Country)

    rate_mean  = [mean(λ_post[:, i])              for i in 1:n_obs]
    rate_ci_lo = [quantile(λ_post[:, i], 0.025)   for i in 1:n_obs]
    rate_ci_hi = [quantile(λ_post[:, i], 0.975)   for i in 1:n_obs]

    yp = Float64.(ypred)
    pp_mean    = [mean(yp[:, i])                  for i in 1:n_obs]
    pp_median  = [median(yp[:, i])                for i in 1:n_obs]
    pp_ci_lo   = [quantile(yp[:, i], 0.025)       for i in 1:n_obs]
    pp_ci_hi   = [quantile(yp[:, i], 0.975)       for i in 1:n_obs]

    df_out = DataFrame(
        country    = countries,
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
    fname = "results/walkfree/tables/country_summary_$(label_str).csv"
    CSV.write(fname, df_out)
    println("  Saved: $fname")
    return df_out
end

tbl_marg = country_table(ypred_marg, λ_post_marg, "marg_gibbs", df_clean, y, P)
tbl_rj   = country_table(ypred_rj,   λ_post_rj,   "rjmcmc",     df_clean, y, P)

# ============================================================================
# 12. Per-country posterior predictive density plots (4×3 grids)
# ============================================================================

println("\n[Section 12] Plotting per-country posterior predictive distributions...")

countries_all = String.(df_clean.Country)
n_per_page    = 12   # 4 rows × 3 columns
n_pages       = ceil(Int, n / n_per_page)

for page in 1:n_pages
    idx_start = (page - 1) * n_per_page + 1
    idx_end   = min(page * n_per_page, n)
    idx_page  = idx_start:idx_end

    panels = Any[]
    for i in idx_page
        yobs    = Float64(y[i])
        yp_marg = Float64.(ypred_marg[:, i])
        yp_rj   = Float64.(ypred_rj[:, i])

        p = plot(title=countries_all[i], titlefontsize=6,
                 xlabel="", ylabel="", legend=false, framestyle=:box,
                 tickfontsize=5)
        density!(p, yp_marg; color=colors[1], linewidth=1.2, alpha=0.8)
        density!(p, yp_rj;   color=colors[2], linewidth=1.2, alpha=0.8)
        vline!(p, [yobs]; color=:black, linewidth=1.5, linestyle=:dash)
        push!(panels, p)
    end

    # Pad incomplete last page with blank panels
    while length(panels) < n_per_page
        push!(panels, plot(axis=false, grid=false, framestyle=:none))
    end

    p_grid = plot(panels..., layout=(4, 3), size=(1000, 1100),
                  plot_title=@sprintf("PP by country (page %d/%d) — Marg-Gibbs (blue), RJMCMC (orange), observed (dashed)", page, n_pages),
                  plot_titlefontsize=8)
    fname = @sprintf("results/walkfree/figures/ppc_countries_%03d.png", page)
    savefig(p_grid, fname)
    println("  Saved: $fname  (countries $(idx_start)-$(idx_end) of $n)")
end

println("  Per-country PPC plots complete.")

println("\nDone.")







# plot alpha and the prior
density(res_marg.α; label="Posterior α (Marg-Gibbs)", color=:blue, linewidth=2)
x = collect(0.01:0.01:10.0)
y = pdf(Gamma(1.0, 1/0.01), x)
density!(x, y; label="Gamma(1, 0.01)", color=:red, linewidth=2)
density(x, y; label="Gamma(1, 0.01)", color=:red, linewidth=2)