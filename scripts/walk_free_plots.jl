# ============================================================================
# Walk Free Foundation Global Slavery Index - Figures and Tables
# ============================================================================
#
# Loads chains saved by walk_free_analysis.jl and produces all figures and
# tables. No MCMC is run here; requires only a single process.
#
# Run after walk_free_analysis.jl:
#   julia --project scripts/walk_free_plots.jl
# ============================================================================

ENV["GKSwstype"] = "100"   # headless GR backend

using DDCRP
using CSV, DataFrames, Distances, LinearAlgebra
using Statistics, StatsBase
using Plots, StatsPlots
using JLD2, XLSX, Printf, Random, Distributions, SpecialFunctions

Random.seed!(42)

# ============================================================================
# Section 0 – Output directories
# ============================================================================

mkpath("results/walkfree/figures")
mkpath("results/walkfree/poisson/figures")
mkpath("results/walkfree/poisson/tables")
mkpath("results/walkfree/nb_grid/figures")
mkpath("results/walkfree/nb_best/figures")
mkpath("results/walkfree/nb_best/tables")
mkpath("results/walkfree/rjmcmc/figures")
mkpath("results/walkfree/rjmcmc_elpd/figures")

# ============================================================================
# Section 1 – Load and preprocess data
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

filter!(x -> x.pop > 0, df)

covariate_cols = [:governance_issues, :lack_basic_needs, :inequality,
                  :disenfranchised_groups, :effects_of_conflict]
required_cols  = [:est_num_ms, :pop, covariate_cols...]
df_clean = dropmissing(df, required_cols)

y = Int.(df_clean.est_num_ms)
P = Int.(df_clean.pop)
n = length(y)

X     = Float64.(Matrix(df_clean[:, covariate_cols]))
X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)

F_svd    = svd(X_std)
pc1      = F_svd.U[:, 1] .* F_svd.S[1]
pc2      = F_svd.U[:, 2] .* F_svd.S[2]
var_expl = 100 .* F_svd.S .^ 2 ./ sum(F_svd.S .^ 2)
pct1     = round(var_expl[1], digits=1)
pct2     = round(var_expl[2], digits=1)

y_lo = 10.0 ^ floor(log10(minimum(y)))
y_hi = 10.0 ^ ceil(log10(maximum(y)))

println("  n = $n countries loaded")

# ============================================================================
# Section 2 – Load saved results
# ============================================================================

println("Loading saved chains and model comparison...")

n_burnin       = 25_000
priors_poisson = PoissonPopulationRatesMargPriors(1.0, 0.1)
priors_marg    = NBPopulationRatesMargPriors(1.0, 0.1, 1.0, 0.1)

df_grid = CSV.read("results/walkfree/model_comparison.csv", DataFrame)

best_r = load("results/walkfree/nb_best/chains/samples_best.jld2", "best_r")
println("  Best r = $best_r")

# ============================================================================
# Section 3 – Helper functions
# ============================================================================

"""Draw Poisson PP samples from (n_iter × n_obs) effective-rate matrix."""
function ppc_from_rates(eff_rates::Matrix{Float64})
    n_iter, n_obs = size(eff_rates)
    y_pred = zeros(Int, n_iter, n_obs)
    for s in 1:n_iter
        for i in 1:n_obs
            y_pred[s, i] = rand(Poisson(max(eff_rates[s, i], 1e-10)))
        end
    end
    return y_pred
end

"""Build (n_iter × n_obs) matrix of Poisson log-likelihoods from effective-rate matrix."""
function compute_ll_matrix(y_obs::Vector{Int}, eff_rates::Matrix{Float64})
    n_iter, n_obs = size(eff_rates)
    ll = zeros(Float64, n_iter, n_obs)
    for i in 1:n_obs
        yi = Float64(y_obs[i])
        lg = loggamma(yi + 1.0)
        for s in 1:n_iter
            μ = max(eff_rates[s, i], 1e-300)
            ll[s, i] = -μ + yi * log(μ) - lg
        end
    end
    return ll
end

"""PPC scatter panel: PP mean vs observed, with 95% CI bars, on log-log scale."""
function ppc_scatter_panel(ypred::Matrix{Int}, y_obs::Vector{Int}, col, title_str)
    pp_mean = vec(mean(Float64.(ypred), dims=1))
    pp_lo   = [quantile(Float64.(ypred[:, i]), 0.025) for i in 1:length(y_obs)]
    pp_hi   = [quantile(Float64.(ypred[:, i]), 0.975) for i in 1:length(y_obs)]
    yf      = Float64.(y_obs)

    p = plot(title=title_str, xlabel="Observed y", ylabel="PP mean",
             xscale=:log10, yscale=:log10, legend=false)
    for i in eachindex(y_obs)
        plot!(p, [yf[i], yf[i]], [max(1.0, pp_lo[i]), max(1.0, pp_hi[i])];
              color=col, alpha=0.3, linewidth=0.8)
    end
    scatter!(p, yf, max.(1.0, pp_mean); color=col, markersize=3,
             markerstrokewidth=0.3, alpha=0.7)
    xy_range = [max(1.0, minimum(yf)), maximum(yf)]
    plot!(p, xy_range, xy_range; color=:black, linestyle=:dash, linewidth=1)
    return p
end

"""Save K, α, s traces/densities, co-clustering heatmap, and logpost trace."""
function save_standard_figures(c_post, α_post, s_post, logpost, prefix, label, color)
    k_post = calculate_n_clusters(c_post)

    p_kt = plot(title="K trace — $label", xlabel="Iteration", ylabel="K")
    plot!(p_kt, k_post, color=color, alpha=0.7, linewidth=0.8, label="")
    savefig(p_kt, "$(prefix)_k_trace.png")

    cm_k = countmap(k_post)
    ks   = sort(collect(keys(cm_k)))
    ps   = [cm_k[k] / length(k_post) for k in ks]
    p_kd = plot(title="K distribution — $label", xlabel="K", ylabel="Probability")
    plot!(p_kd, ks, ps, markershape=:circle, linewidth=2, color=color, label="")
    savefig(p_kd, "$(prefix)_k_distribution.png")

    p_αt = plot(title="α trace — $label", xlabel="Iteration", ylabel="α")
    plot!(p_αt, α_post, color=color, alpha=0.7, linewidth=0.8, label="")
    savefig(p_αt, "$(prefix)_alpha_trace.png")

    p_αd = plot(title="α density — $label", xlabel="α", ylabel="Density")
    density!(p_αd, α_post, color=color, linewidth=2, label="")
    savefig(p_αd, "$(prefix)_alpha_density.png")

    p_st = plot(title="s trace — $label", xlabel="Iteration", ylabel="s (decay scale)")
    plot!(p_st, s_post, color=color, alpha=0.7, linewidth=0.8, label="")
    savefig(p_st, "$(prefix)_s_trace.png")

    p_sd = plot(title="s density — $label", xlabel="s (decay scale)", ylabel="Density")
    density!(p_sd, s_post, color=color, linewidth=2, label="")
    savefig(p_sd, "$(prefix)_s_density.png")

    sim = compute_similarity_matrix(c_post)
    p_sim = heatmap(sim, title="Co-clustering: $label", xlabel="Country", ylabel="Country",
                    color=:viridis, colorbar_title="Pr(same cluster)", aspect_ratio=:equal)
    savefig(p_sim, "$(prefix)_coclustering.png")

    p_lp = plot(title="Log-posterior trace — $label", xlabel="Iteration", ylabel="Log-posterior")
    plot!(p_lp, logpost, color=color, alpha=0.7, linewidth=0.8, label="")
    savefig(p_lp, "$(prefix)_logpost_trace.png")

    return k_post, sim
end

"""Sample ρ_k from Gamma posterior for each PoissonPopMarg iteration;
return (n_iter × n_obs) effective-rate matrix P[i] * ρ_{k(i)}."""
function poisson_rate_samples(c_post::Matrix{Int}, y_obs::Vector{Int},
                               P_obs::Vector{Int}, priors_p)
    n_iter, n_obs = size(c_post)
    eff = zeros(Float64, n_iter, n_obs)
    for s in 1:n_iter
        for tbl in table_vector(c_post[s, :])
            S_k = Float64(sum(y_obs[tbl]))
            Pk  = Float64(sum(P_obs[tbl]))
            ρ_k = rand(Gamma(S_k + priors_p.ρ_a, 1.0 / (Pk + priors_p.ρ_b)))
            for i in tbl
                eff[s, i] = Float64(P_obs[i]) * ρ_k
            end
        end
    end
    return eff
end

# ============================================================================
# Section 4 – Poisson figures
# ============================================================================

println("\n[Section 4] Poisson figures")
@load "results/walkfree/poisson/chains/samples_poisson.jld2" samples_poisson diag_poisson

idx_p    = (n_burnin + 1):size(samples_poisson.c, 1)
c_post_p = samples_poisson.c[idx_p, :]
α_post_p = samples_poisson.α_ddcrp[idx_p]
s_post_p = samples_poisson.s_ddcrp[idx_p]

k_post_p, sim_p = save_standard_figures(
    c_post_p, α_post_p, s_post_p,
    samples_poisson.logpost[idx_p],
    "results/walkfree/poisson/figures/poisson",
    "Poisson", :forestgreen
)

println("  Computing Poisson PPC...")
eff_rates_p = poisson_rate_samples(c_post_p, y, P, priors_poisson)
ypred_p     = ppc_from_rates(eff_rates_p)
ll_mat_p    = compute_ll_matrix(y, eff_rates_p)
waic_p      = compute_waic(y, eff_rates_p)
loo_p       = compute_psis_loo(ll_mat_p)
println("  WAIC=$(round(waic_p.waic, digits=2))  ELPD-LOO=$(round(loo_p.elpd_loo, digits=2))")

p_ppc_p_dens = plot(title="Posterior predictive check — Poisson", xlabel="log₁₀(y+1)", ylabel="Density")
density!(p_ppc_p_dens, vec(log10.(Float64.(ypred_p) .+ 1)); label="PP", color=:forestgreen, linewidth=1.5)
density!(p_ppc_p_dens, log10.(Float64.(y) .+ 1); label="Observed", color=:black, linewidth=2, linestyle=:dash)
savefig(p_ppc_p_dens, "results/walkfree/poisson/figures/ppc_density.png")

p_ppc_p_scatter = ppc_scatter_panel(ypred_p, y, :forestgreen, "PP mean vs observed — Poisson")
savefig(p_ppc_p_scatter, "results/walkfree/poisson/figures/ppc_scatter.png")

let
    rate_mean = [mean(eff_rates_p[:, i] ./ Float64(P[i]))            for i in 1:n]
    rate_lo   = [quantile(eff_rates_p[:, i] ./ Float64(P[i]), 0.025) for i in 1:n]
    rate_hi   = [quantile(eff_rates_p[:, i] ./ Float64(P[i]), 0.975) for i in 1:n]
    yp        = Float64.(ypred_p)
    df_p = DataFrame(
        country    = String.(df_clean.Country),
        population = P,
        y_observed = y,
        rate_mean  = round.(rate_mean, digits=6),
        rate_ci_lo = round.(rate_lo,   digits=6),
        rate_ci_hi = round.(rate_hi,   digits=6),
        pp_mean    = round.([mean(yp[:, i]) for i in 1:n],            digits=1),
        pp_median  = round.([median(yp[:, i]) for i in 1:n],          digits=1),
        pp_ci_lo   = round.([quantile(yp[:, i], 0.025) for i in 1:n], digits=1),
        pp_ci_hi   = round.([quantile(yp[:, i], 0.975) for i in 1:n], digits=1),
        waic_i     = round.(waic_p.waic_i, digits=4),
        elpd_loo_i = round.(loo_p.loo_i,   digits=4),
        k_hat      = round.(loo_p.k_hat,   digits=4),
    )
    sort!(df_p, :y_observed, rev=true)
    CSV.write("results/walkfree/poisson/tables/country_summary_poisson.csv", df_p)
    println("  Poisson country table saved.")
end

println("  Poisson figures complete.")

# ============================================================================
# Section 5 – Grid summary figures
# ============================================================================

println("\n[Section 5] Grid summary figures")

nb_rows     = filter(r -> r.model != "Poisson", df_grid)
poisson_row = filter(r -> r.model == "Poisson", df_grid)[1, :]
r_vals      = nb_rows.r_fixed

p_waic_r = plot(r_vals, nb_rows.waic;
                markershape=:circle, linewidth=2, color=:steelblue,
                title="WAIC vs r", xlabel="r (fixed)", ylabel="WAIC", legend=:topright)
hline!(p_waic_r, [poisson_row.waic]; color=:forestgreen, linestyle=:dash, label="Poisson")
savefig(p_waic_r, "results/walkfree/nb_grid/figures/waic_vs_r.png")

p_loo_r = plot(r_vals, nb_rows.elpd_loo;
               markershape=:circle, linewidth=2, color=:steelblue,
               title="ELPD-LOO vs r", xlabel="r (fixed)", ylabel="ELPD-LOO", legend=:topright)
hline!(p_loo_r, [poisson_row.elpd_loo]; color=:forestgreen, linestyle=:dash, label="Poisson")
savefig(p_loo_r, "results/walkfree/nb_grid/figures/elpd_loo_vs_r.png")

p_k_r = plot(r_vals, nb_rows.mean_K;
             markershape=:circle, linewidth=2, color=:steelblue,
             title="Mean K vs r", xlabel="r (fixed)", ylabel="Mean K", legend=:topright)
hline!(p_k_r, [poisson_row.mean_K]; color=:forestgreen, linestyle=:dash, label="Poisson")
savefig(p_k_r, "results/walkfree/nb_grid/figures/mean_k_vs_r.png")

p_metrics = plot(p_waic_r, p_loo_r, p_k_r, layout=(1, 3), size=(1200, 400),
                 plot_title="Model comparison metrics vs r")
savefig(p_metrics, "results/walkfree/nb_grid/figures/metric_vs_r.png")

println("  Grid summary figures saved.")

# ============================================================================
# Section 6 – Best-r NB figures
# ============================================================================

println("\n[Section 6] Best-r NB figures (r=$best_r)")
@load "results/walkfree/nb_best/chains/samples_best.jld2" samples_best diag_best best_r

idx_b    = (n_burnin + 1):size(samples_best.c, 1)
c_post_b = samples_best.c[idx_b, :]
α_post_b = samples_best.α_ddcrp[idx_b]
s_post_b = samples_best.s_ddcrp[idx_b]
λ_post_b = samples_best.λ[idx_b, :]

k_post_b, sim_b = save_standard_figures(
    c_post_b, α_post_b, s_post_b,
    samples_best.logpost[idx_b],
    "results/walkfree/nb_best/figures/nb_best",
    "NB best r=$(best_r)", :steelblue
)

z_map_b  = point_estimate_clustering(c_post_b; method=:MAP)
csizes_b = countmap(z_map_b)

println("  Computing NB best PPC...")
eff_rates_b = λ_post_b .* Float64.(P)'
ypred_b     = ppc_from_rates(eff_rates_b)
ll_mat_b    = compute_ll_matrix(y, eff_rates_b)
waic_b      = compute_waic(y, eff_rates_b)
loo_b       = compute_psis_loo(ll_mat_b)
println("  WAIC=$(round(waic_b.waic, digits=2))  ELPD-LOO=$(round(loo_b.elpd_loo, digits=2))")

p_ppc_b_dens = plot(title="PPC — NB best r=$(best_r)", xlabel="log₁₀(y+1)", ylabel="Density")
density!(p_ppc_b_dens, vec(log10.(Float64.(ypred_b) .+ 1)); label="PP", color=:steelblue, linewidth=1.5)
density!(p_ppc_b_dens, log10.(Float64.(y) .+ 1); label="Observed", color=:black, linewidth=2, linestyle=:dash)
savefig(p_ppc_b_dens, "results/walkfree/nb_best/figures/ppc_density.png")

p_ppc_b_scatter = ppc_scatter_panel(ypred_b, y, :steelblue, "PP mean vs observed — NB r=$(best_r)")
savefig(p_ppc_b_scatter, "results/walkfree/nb_best/figures/ppc_scatter.png")

# Posterior link probability arrows
link_prob_b = zeros(Float64, n, n)
for s in axes(c_post_b, 1)
    for i in 1:n
        link_prob_b[i, c_post_b[s, i]] += 1.0
    end
end
link_prob_b ./= size(c_post_b, 1)

keep_b   = findall(i -> csizes_b[z_map_b[i]] > 1, 1:n)
unique_b = sort(unique(z_map_b[keep_b]))
K_ns_b   = length(unique_b)
remap_b  = Dict(c => k for (k, c) in enumerate(unique_b))
z_ns_b   = [remap_b[z_map_b[i]] for i in keep_b]
cl_b(k)  = [j for (j, z) in enumerate(z_ns_b) if z == k]

if K_ns_b > 0
    link_thresh = 0.05
    p_arrows_b = plot(
        xlabel = "PC1 ($pct1% var explained)",
        ylabel = "PC2 ($pct2% var explained)",
        title  = "Posterior link probabilities — NB r=$(best_r), singletons excluded",
        legend = :outerright, size = (900, 550)
    )
    for i in keep_b, j in keep_b
        i == j && continue
        p_ij = link_prob_b[i, j]
        p_ij < link_thresh && continue
        plot!(p_arrows_b, [pc1[i], pc1[j]], [pc2[i], pc2[j]];
              label="", color=:gray, alpha=min(p_ij * 2, 0.8),
              linewidth=0.5 + 2.5 * p_ij, arrow=true)
    end
    for k in 1:K_ns_b
        ji = cl_b(k)
        scatter!(p_arrows_b, pc1[keep_b[ji]], pc2[keep_b[ji]];
                 label="Cluster $k", markersize=5, markerstrokewidth=0.4)
    end
    savefig(p_arrows_b, "results/walkfree/nb_best/figures/cluster_link_arrows.png")

    p_pc2_b = plot(xlabel="PC1 ($pct1% var explained)", ylabel="PC2 ($pct2% var explained)",
                   title="PCA — MAP clusters (NB r=$(best_r))",
                   legend=:outerright, size=(900, 500))
    for k in 1:K_ns_b
        ji = cl_b(k)
        scatter!(p_pc2_b, pc1[keep_b[ji]], pc2[keep_b[ji]];
                 label="Cluster $k", markersize=5, markerstrokewidth=0.4)
    end
    savefig(p_pc2_b, "results/walkfree/nb_best/figures/cluster_pca.png")

    p_pc1_b = plot(xlabel="PC1 ($pct1% var explained)", ylabel="People in modern slavery",
                   title="PC1 vs Count — MAP clusters (NB r=$(best_r))",
                   legend=:outerright, yscale=:log10, ylims=(y_lo, y_hi), size=(900, 500))
    for k in 1:K_ns_b
        ji = cl_b(k)
        scatter!(p_pc1_b, pc1[keep_b[ji]], Float64.(y[keep_b[ji]]);
                 label="Cluster $k", markersize=5, markerstrokewidth=0.4)
    end
    savefig(p_pc1_b, "results/walkfree/nb_best/figures/cluster_pc1_vs_count.png")
end

# Country-level table
let
    rate_mean = [mean(λ_post_b[:, i])            for i in 1:n]
    rate_lo   = [quantile(λ_post_b[:, i], 0.025) for i in 1:n]
    rate_hi   = [quantile(λ_post_b[:, i], 0.975) for i in 1:n]
    yp        = Float64.(ypred_b)
    df_b = DataFrame(
        country    = String.(df_clean.Country),
        population = P,
        y_observed = y,
        rate_mean  = round.(rate_mean,  digits=6),
        rate_ci_lo = round.(rate_lo,    digits=6),
        rate_ci_hi = round.(rate_hi,    digits=6),
        pp_mean    = round.([mean(yp[:, i]) for i in 1:n],            digits=1),
        pp_median  = round.([median(yp[:, i]) for i in 1:n],          digits=1),
        pp_ci_lo   = round.([quantile(yp[:, i], 0.025) for i in 1:n], digits=1),
        pp_ci_hi   = round.([quantile(yp[:, i], 0.975) for i in 1:n], digits=1),
        waic_i     = round.(waic_b.waic_i, digits=4),
        elpd_loo_i = round.(loo_b.loo_i,   digits=4),
        k_hat      = round.(loo_b.k_hat,   digits=4),
    )
    sort!(df_b, :y_observed, rev=true)
    CSV.write("results/walkfree/nb_best/tables/country_summary_nb_best.csv", df_b)
    println("  Country table saved.")
end

# Comparison: Poisson vs NB best — per-country marginal PPC
# Sort countries by observed count (descending)
ord       = sortperm(y, rev=true)
xs        = 1:n
y_sorted  = Float64.(y[ord])

pp_mean_p = [mean(Float64.(ypred_p[:, i])) for i in ord]
pp_lo_p   = [quantile(Float64.(ypred_p[:, i]), 0.025) for i in ord]
pp_hi_p   = [quantile(Float64.(ypred_p[:, i]), 0.975) for i in ord]

pp_mean_b = [mean(Float64.(ypred_b[:, i])) for i in ord]
pp_lo_b   = [quantile(Float64.(ypred_b[:, i]), 0.025) for i in ord]
pp_hi_b   = [quantile(Float64.(ypred_b[:, i]), 0.975) for i in ord]

p_cmp_dens = plot(
    yscale  = :log10,
    xlabel  = "Country (sorted by observed count, descending)",
    ylabel  = "Count",
    title   = "Per-country PPC — Poisson vs NB r=$(best_r)",
    legend  = :topright,
    size    = (1200, 500),
)
# Poisson 95% CI bars
for i in xs
    plot!(p_cmp_dens, [i, i], [max(pp_lo_p[i], 1.0), max(pp_hi_p[i], 1.0)];
          color=:forestgreen, alpha=0.25, linewidth=1.5, label="")
end
# NB 95% CI bars
for i in xs
    plot!(p_cmp_dens, [i, i], [max(pp_lo_b[i], 1.0), max(pp_hi_b[i], 1.0)];
          color=:steelblue, alpha=0.25, linewidth=1.5, label="")
end
# Poisson posterior means
scatter!(p_cmp_dens, xs, max.(pp_mean_p, 1.0);
         color=:forestgreen, markersize=2.5, markerstrokewidth=0,
         alpha=0.8, label="Poisson mean")
# NB posterior means
scatter!(p_cmp_dens, xs, max.(pp_mean_b, 1.0);
         color=:steelblue, markersize=2.5, markerstrokewidth=0,
         alpha=0.8, label="NB r=$(best_r) mean")
# Observed counts
scatter!(p_cmp_dens, xs, max.(y_sorted, 1.0);
         color=:black, markersize=3, markerstrokewidth=0,
         markershape=:diamond, alpha=0.9, label="Observed")
savefig(p_cmp_dens, "results/walkfree/figures/ppc_comparison.png")

p_cmp_k = plot(title="K distribution — Poisson vs NB r=$(best_r)", xlabel="K", ylabel="Probability")
let cm = countmap(k_post_p); ks = sort(collect(keys(cm)))
    plot!(p_cmp_k, ks, [cm[k]/length(k_post_p) for k in ks]; label="Poisson", color=:forestgreen, markershape=:circle, linewidth=2)
end
let cm = countmap(k_post_b); ks = sort(collect(keys(cm)))
    plot!(p_cmp_k, ks, [cm[k]/length(k_post_b) for k in ks]; label="NB r=$(best_r)", color=:steelblue, markershape=:circle, linewidth=2)
end
savefig(p_cmp_k, "results/walkfree/figures/k_distribution_comparison.png")

println("  NB best figures complete.")

# ============================================================================
# Section 7 – RJMCMC validation figures
# ============================================================================

println("\n[Section 7] RJMCMC validation figures")
@load "results/walkfree/rjmcmc/chains/samples_rjmcmc.jld2" samples_rj diag_rj

idx_rj    = (n_burnin + 1):size(samples_rj.c, 1)
c_post_rj = samples_rj.c[idx_rj, :]
α_post_rj = samples_rj.α_ddcrp[idx_rj]
s_post_rj = samples_rj.s_ddcrp[idx_rj]

k_post_rj, sim_rj = save_standard_figures(
    c_post_rj, α_post_rj, s_post_rj,
    samples_rj.logpost[idx_rj],
    "results/walkfree/rjmcmc/figures/rjmcmc",
    "RJMCMC r=$(best_r)", :darkorange
)

p_k_overlay = plot(title="K distribution — Marg-Gibbs vs RJMCMC (r=$(best_r))",
                   xlabel="K", ylabel="Probability")
let cm = countmap(k_post_b); ks = sort(collect(keys(cm)))
    plot!(p_k_overlay, ks, [cm[k]/length(k_post_b) for k in ks];
          label="Marg-Gibbs", color=:steelblue, markershape=:circle, linewidth=2)
end
let cm = countmap(k_post_rj); ks = sort(collect(keys(cm)))
    plot!(p_k_overlay, ks, [cm[k]/length(k_post_rj) for k in ks];
          label="RJMCMC", color=:darkorange, markershape=:square, linewidth=2)
end
savefig(p_k_overlay, "results/walkfree/figures/k_distribution_marg_vs_rjmcmc.png")

p_α_ov = plot(title="α density — Marg-Gibbs vs RJMCMC (r=$(best_r))", xlabel="α", ylabel="Density")
density!(p_α_ov, α_post_b;  label="Marg-Gibbs", color=:steelblue, linewidth=2)
density!(p_α_ov, α_post_rj; label="RJMCMC",    color=:darkorange, linewidth=2)
savefig(p_α_ov, "results/walkfree/figures/alpha_density_marg_vs_rjmcmc.png")

p_s_ov = plot(title="s density — Marg-Gibbs vs RJMCMC (r=$(best_r))",
              xlabel="s (decay scale)", ylabel="Density")
density!(p_s_ov, s_post_b;  label="Marg-Gibbs", color=:steelblue, linewidth=2)
density!(p_s_ov, s_post_rj; label="RJMCMC",    color=:darkorange, linewidth=2)
savefig(p_s_ov, "results/walkfree/figures/s_density_marg_vs_rjmcmc.png")

println("  RJMCMC validation figures saved.")

# ============================================================================
# Section 8 – RJMCMC best ELPD figures
# ============================================================================

println("\n[Section 8] RJMCMC best ELPD-LOO figures")
@load "results/walkfree/rjmcmc_elpd/chains/samples_rjmcmc.jld2" samples_best_rj diag_best_rj best_by_elpd

idx_best    = (n_burnin + 1):size(samples_best_rj.c, 1)
c_post_best = samples_best_rj.c[idx_best, :]
α_post_best = samples_best_rj.α_ddcrp[idx_best]
s_post_best = samples_best_rj.s_ddcrp[idx_best]

save_standard_figures(c_post_best, α_post_best, s_post_best,
    samples_best_rj.logpost[idx_best],
    "results/walkfree/rjmcmc_elpd/figures/rjmcmc",
    "RJMCMC r=$(best_by_elpd)", :darkorange)

println("  RJMCMC best ELPD figures saved.")

# ============================================================================
# Paper figures
# ============================================================================

println("\n[Paper figures] Copying to results/walkfree/plots_for_paper/")
mkpath("results/walkfree/plots_for_paper")

paper_figures = [
    "results/walkfree/nb_grid/figures/elpd_loo_vs_r.png",
    "results/walkfree/nb_best/figures/nb_best_k_distribution.png",
    "results/walkfree/nb_best/figures/nb_best_alpha_density.png",
    "results/walkfree/nb_best/figures/nb_best_s_density.png",
    "results/walkfree/figures/k_distribution_marg_vs_rjmcmc.png",
    "results/walkfree/figures/alpha_density_marg_vs_rjmcmc.png",
    "results/walkfree/figures/s_density_marg_vs_rjmcmc.png",
    "results/walkfree/nb_best/figures/cluster_pca.png",
    "results/walkfree/figures/ppc_comparison.png",
    "results/walkfree/nb_best/figures/nb_best_coclustering.png",
]

for src in paper_figures
    dst = joinpath("results/walkfree/plots_for_paper", basename(src))
    cp(src, dst; force=true)
    println("  $(basename(src))")
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^60)
println("All figures and tables generated.")
println("\nPaper figures in: results/walkfree/plots_for_paper/")
println("Key output files:")
println("  results/walkfree/nb_grid/figures/metric_vs_r.png")
println("  results/walkfree/figures/ppc_comparison.png")
println("  results/walkfree/figures/k_distribution_comparison.png")
println("  results/walkfree/figures/k_distribution_marg_vs_rjmcmc.png")
println("  results/walkfree/nb_best/figures/cluster_pca.png")
println("  results/walkfree/nb_best/tables/country_summary_nb_best.csv")
println("="^60)
