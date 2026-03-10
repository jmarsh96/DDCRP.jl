# ============================================================================
# Walk Free Foundation Global Slavery Index - ddCRP Analysis
# ============================================================================
#
# Structure:
#   1. Poisson model — establishes baseline (tends to over-estimate K)
#   2. NB grid search over r ∈ {0.5, 1.0, ..., 20.0} with r fixed (parallel)
#   3. Model comparison: WAIC, CRPS, ELPD-LOO, mean K
#   4. Best-r NB post-processing (extended diagnostics)
#   5. RJMCMC confirmation at best r
#
# CLI options:
#   --test         reduced iteration counts (smoke test)
#   --slurm        use SlurmClusterManager instead of local workers
#   --nprocs N     number of local worker processes (default: CPU_THREADS ÷ 2)
# ============================================================================

using Distributed

# ── CLI args (parsed before addprocs) ─────────────────────────────────────────
const TEST_RUN  = "--test"  in ARGS || get(ENV, "TEST_RUN", "false") == "true"
const USE_SLURM = "--slurm" in ARGS
const N_PROCS   = let idx = findfirst(==("--nprocs"), ARGS)
    isnothing(idx) ? max(1, Sys.CPU_THREADS ÷ 2) : parse(Int, ARGS[idx + 1])
end

if TEST_RUN
    println("*** TEST RUN MODE — reduced iteration counts ***")
end

# ── Add workers ───────────────────────────────────────────────────────────────
if USE_SLURM
    using SlurmClusterManager
    addprocs(SlurmManager())
    println("Added $(nworkers()) SLURM workers")
else
    addprocs(N_PROCS)
    println("Added $(nworkers()) local workers")
end
println("Running with $(nworkers()) worker(s)")

# ── Load packages on all processes ────────────────────────────────────────────
@everywhere begin
    ENV["GKSwstype"] = "100"   # headless Plots (required on worker nodes)
    using DDCRP
    using CSV, DataFrames, Distances
    using Statistics, StatsBase, LinearAlgebra
    using Plots, StatsPlots
    using Random
    using Printf
    using XLSX
    using JLD2
    using Distributions
    using SpecialFunctions
end

Random.seed!(2025)

# ============================================================================
# Section 0 – Output directories
# ============================================================================

mkpath("results/walkfree")
mkpath("results/walkfree/figures")
mkpath("results/walkfree/chains")
mkpath("results/walkfree/poisson/figures")
mkpath("results/walkfree/poisson/chains")
mkpath("results/walkfree/poisson/tables")
mkpath("results/walkfree/nb_grid/figures")
mkpath("results/walkfree/nb_grid/chains")
mkpath("results/walkfree/nb_best/figures")
mkpath("results/walkfree/nb_best/chains")
mkpath("results/walkfree/nb_best/tables")
mkpath("results/walkfree/rjmcmc/figures")
mkpath("results/walkfree/rjmcmc/chains")

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
println("  Countries after dropping missing: $(nrow(df_clean)) / $(nrow(df))")

y = Int.(df_clean.est_num_ms)
P = Int.(df_clean.pop)

X     = Float64.(Matrix(df_clean[:, covariate_cols]))
X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)
D     = pairwise(Euclidean(), X_std', dims=2)

let dists = [D[i,j] for i in axes(D,1) for j in (i+1):last(axes(D,2))]
    display(histogram(dists, xlabel="Distance", ylabel="Count",
                      title="Pairwise Covariate Distances (upper tri)", legend=false))
end

n = length(y)
println("  n = $n observations")
println("  y range: [$(minimum(y)), $(maximum(y))]")
println("  P range: [$(minimum(P)), $(maximum(P))]")

D_offdiag = [D[i,j] for i in 1:n, j in 1:n if i != j]
println("  D (off-diagonal) mean=$(round(mean(D_offdiag),digits=3))  " *
        "median=$(round(median(D_offdiag),digits=3))  max=$(round(maximum(D_offdiag),digits=3))")
for s_test in [0.1, 0.5, 1.0, 2.0, 5.0]
    pct = round(100 * mean(exp.(-s_test .* D_offdiag) .> 0.01), digits=1)
    println("    s=$s_test: $(pct)% of pairs have decay > 0.01")
end

F_svd    = svd(X_std)
pc1      = F_svd.U[:, 1] .* F_svd.S[1]
pc2      = F_svd.U[:, 2] .* F_svd.S[2]
var_expl = 100 .* F_svd.S .^ 2 ./ sum(F_svd.S .^ 2)
pct1     = round(var_expl[1], digits=1)
pct2     = round(var_expl[2], digits=1)

y_lo = 10.0 ^ floor(log10(minimum(y)))
y_hi = 10.0 ^ ceil(log10(maximum(y)))

# ============================================================================
# Section 2 – Priors and MCMC options
# ============================================================================

# NB population priors: γ ~ Gamma(1, rate=0.1), r ~ Gamma(1, rate=0.1)
priors_marg   = NBPopulationRatesMargPriors(1.0, 0.1, 1.0, 0.1)
priors_unmarg = NBPopulationRatesPriors(1.0, 0.1, 1.0, 0.1)
priors_poisson = PoissonPopulationRatesMargPriors(1.0, 0.1)

# α prior: Gamma(1, 0.01) — diffuse, E[α]=100
# s prior: Gamma(1, 0.01) — diffuse, E[s]=100
ddcrp_params = DDCRPParams(1.0, 1.0, 1.0, 0.01, 1.0, 0.01)

if TEST_RUN
    n_samples = 500
    n_burnin  = 100
    r_grid    = [1.0, 5.0, 10.0]   # minimal grid for smoke test
else
    n_samples = 30_000
    n_burnin  = 5_000
    r_grid    = vcat(0.5:0.5:20.0) # 40 values
end

n_burnin < n_samples || error("n_burnin ($n_burnin) must be less than n_samples ($n_samples)")

opts_nb = MCMCOptions(
    n_samples         = n_samples,
    verbose           = false,
    track_diagnostics = true,
    infer_params      = Dict(:r => false, :c => true, :α_ddcrp => true, :s_ddcrp => true),
    prop_sds          = Dict(:s_ddcrp => 0.3)
)

opts_poisson = MCMCOptions(
    n_samples         = n_samples,
    verbose           = false,
    track_diagnostics = true,
    infer_params      = Dict(:c => true, :α_ddcrp => true, :s_ddcrp => true),
    prop_sds          = Dict(:s_ddcrp => 0.3)
)

# ============================================================================
# Section 3 – Helper functions (defined @everywhere for use by grid workers)
# ============================================================================

@everywhere begin

"""Draw Poisson PP samples from (n_iter × n_obs) effective-rate matrix."""
function ppc_from_rates(eff_rates::Matrix{Float64}, P::Vector{Int})
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

"""CRPS via the sorted-sample estimator. Returns (per-obs CRPS vector, mean CRPS)."""
function compute_crps(y_obs::Vector{Int}, y_pred::Matrix{Int})
    n_iter, n_obs = size(y_pred)
    crps_i = zeros(n_obs)
    for i in 1:n_obs
        yi = Float64(y_obs[i])
        yp = sort(Float64.(y_pred[:, i]))
        e_ay = mean(abs.(yp .- yi))
        s = 0.0
        for k in 1:n_iter
            s += (2k - n_iter - 1) * yp[k]
        end
        e_aa = s / (n_iter * (n_iter - 1))
        crps_i[i] = e_ay - e_aa
    end
    return (crps_i, mean(crps_i))
end

"""Mean absolute loss: mean over obs of |pp_mean_i - y_obs_i|."""
function compute_mal(y_obs::Vector{Int}, y_pred::Matrix{Int})
    n_obs = length(y_obs)
    mal = 0.0
    for i in 1:n_obs
        mal += abs(mean(Float64.(y_pred[:, i])) - Float64(y_obs[i]))
    end
    return mal / n_obs
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

"""
Run one r-value of the NB grid search. Called via pmap.
Saves chain + figures; returns a metrics NamedTuple.
"""
function run_grid_r(r_fixed, y, P, D, n_burnin, ddcrp_params, priors_marg, opts_nb)
    r_tag = replace(@sprintf("%.1f", r_fixed), "." => "p")
    r_dir = "results/walkfree/nb_grid/r$(r_tag)"
    mkpath("$(r_dir)/figures")
    mkpath("$(r_dir)/chains")

    t_start = time()
    samples_r, diag_r = mcmc(
        NBPopulationRatesMarg(), y, P, D, ddcrp_params, priors_marg, ConjugateProposal();
        opts        = opts_nb,
        init_params = Dict{Symbol,Any}(:r => Float64(r_fixed))
    )
    t_elapsed = time() - t_start

    chain_file = "$(r_dir)/chains/samples.jld2"
    @save chain_file samples_r diag_r r_fixed

    idx_r    = (n_burnin + 1):size(samples_r.c, 1)
    c_post_r = samples_r.c[idx_r, :]
    α_post_r = samples_r.α_ddcrp[idx_r]
    s_post_r = samples_r.s_ddcrp[idx_r]
    λ_post_r = samples_r.λ[idx_r, :]
    k_post_r = calculate_n_clusters(c_post_r)

    save_standard_figures(c_post_r, α_post_r, s_post_r,
        samples_r.logpost[idx_r],
        "$(r_dir)/figures/r$(r_tag)",
        "NB r=$(r_fixed)", :steelblue)

    z_map_r = point_estimate_clustering(c_post_r; method=:MAP)

    eff_rates_r = λ_post_r .* Float64.(P)'
    ypred_r     = ppc_from_rates(eff_rates_r, P)
    ll_mat_r    = compute_ll_matrix(y, eff_rates_r)
    waic_r      = compute_waic(y, eff_rates_r)
    loo_r       = compute_psis_loo(ll_mat_r)
    crps_r      = compute_crps(y, ypred_r)
    mal_r       = compute_mal(y, ypred_r)

    p_ppc_r = ppc_scatter_panel(ypred_r, y, :steelblue, "PP mean vs observed — r=$(r_fixed)")
    savefig(p_ppc_r, "$(r_dir)/figures/ppc_scatter.png")

    return (
        r         = r_fixed,
        mean_K    = mean(k_post_r),
        median_K  = median(k_post_r),
        mode_K    = argmax(countmap(k_post_r)),
        mean_α    = mean(α_post_r),
        mean_s    = mean(s_post_r),
        ess_K     = effective_sample_size(Float64.(k_post_r)),
        waic      = waic_r.waic,
        lppd      = waic_r.lppd,
        p_waic    = waic_r.p_waic,
        elpd_loo  = loo_r.elpd_loo,
        k_hat_bad = sum(loo_r.k_hat .> 0.7),
        crps      = crps_r[2],
        mal       = mal_r,
        time_s    = t_elapsed,
        z_map     = z_map_r,
    )
end

end # @everywhere

# Main-process-only helpers (not needed by grid workers)

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

"""Print MAP cluster membership to stdout."""
function print_map_clusters(c_post::Matrix{Int}, df_clean::DataFrame, label::String)
    z_map  = point_estimate_clustering(c_post; method=:MAP)
    csizes = countmap(z_map)
    println("\n=== MAP cluster membership: $label ===")
    for (cl, cnt) in sort(collect(csizes), by=x -> x[2])
        members = sort(String.(df_clean.Country[findall(==(cl), z_map)]))
        println("  Cluster $cl (n=$cnt): " * join(members, ", "))
    end
    return z_map, csizes
end

# ============================================================================
# Section 4 – Poisson model (main process)
# ============================================================================

println("\n[Section 4] PoissonPopulationRatesMarg – Gibbs (ConjugateProposal)")
samples_poisson, diag_poisson = mcmc(
    PoissonPopulationRatesMarg(),
    y, P, D,
    ddcrp_params,
    priors_poisson,
    ConjugateProposal();
    opts = opts_poisson
)
println("  Total time: $(round(diag_poisson.total_time, digits=1)) s")
@save "results/walkfree/poisson/chains/samples_poisson.jld2" samples_poisson diag_poisson ddcrp_params priors_poisson
println("  Chain saved.")

idx_p    = (n_burnin + 1):size(samples_poisson.c, 1)
c_post_p = samples_poisson.c[idx_p, :]
α_post_p = samples_poisson.α_ddcrp[idx_p]
s_post_p = samples_poisson.s_ddcrp[idx_p]

println("\n--- PoissonPopulationRatesMarg (post burn-in = $(length(idx_p)) samples) ---")
k_post_p = calculate_n_clusters(c_post_p)
println("  K:  mean=$(round(mean(k_post_p), digits=2))  median=$(median(k_post_p))  mode=$(argmax(countmap(k_post_p)))")
println("  α:  mean=$(round(mean(α_post_p), digits=3))  95%CI=[$(round(quantile(α_post_p,0.025),digits=3)), $(round(quantile(α_post_p,0.975),digits=3))]")
println("  s:  mean=$(round(mean(s_post_p), digits=3))  95%CI=[$(round(quantile(s_post_p,0.025),digits=3)), $(round(quantile(s_post_p,0.975),digits=3))]")
println("  ESS(K)=$(round(effective_sample_size(Float64.(k_post_p)), digits=1))  ESS(α)=$(round(effective_sample_size(α_post_p), digits=1))  ESS(s)=$(round(effective_sample_size(s_post_p), digits=1))")

k_post_p, sim_p = save_standard_figures(
    c_post_p, α_post_p, s_post_p,
    samples_poisson.logpost[idx_p],
    "results/walkfree/poisson/figures/poisson",
    "Poisson", :forestgreen
)

z_map_p, csizes_p = print_map_clusters(c_post_p, df_clean, "PoissonPopulationRatesMarg")
k_sorted_p = sort(collect(countmap(k_post_p)), by=x -> -x[2])
println("  K posterior (top 10):")
for (k, cnt) in k_sorted_p[1:min(10, length(k_sorted_p))]
    @printf "    K=%-4d  p=%.4f\n" k cnt/length(k_post_p)
end

println("\n[Poisson] Computing predictive metrics...")
eff_rates_p = poisson_rate_samples(c_post_p, y, P, priors_poisson)
ypred_p     = ppc_from_rates(eff_rates_p, P)
ll_mat_p    = compute_ll_matrix(y, eff_rates_p)
waic_p      = compute_waic(y, eff_rates_p)
loo_p       = compute_psis_loo(ll_mat_p)
crps_p      = compute_crps(y, ypred_p)
mal_p       = compute_mal(y, ypred_p)
println("  WAIC=$(round(waic_p.waic, digits=2))  ELPD-LOO=$(round(loo_p.elpd_loo, digits=2))  CRPS=$(round(crps_p[2], digits=4))  MAL=$(round(mal_p, digits=4))")
println("  k̂ > 0.7 for $(sum(loo_p.k_hat .> 0.7)) / $n observations")

p_ppc_p_dens = plot(title="Posterior predictive check — Poisson", xlabel="log₁₀(y+1)", ylabel="Density")
density!(p_ppc_p_dens, vec(log10.(Float64.(ypred_p) .+ 1)); label="PP", color=:forestgreen, linewidth=1.5)
density!(p_ppc_p_dens, log10.(Float64.(y) .+ 1); label="Observed", color=:black, linewidth=2, linestyle=:dash)
savefig(p_ppc_p_dens, "results/walkfree/poisson/figures/ppc_density.png")

p_ppc_p_scatter = ppc_scatter_panel(ypred_p, y, :forestgreen, "PP mean vs observed — Poisson")
savefig(p_ppc_p_scatter, "results/walkfree/poisson/figures/ppc_scatter.png")

let
    rate_mean = [mean(eff_rates_p[:, i] ./ Float64(P[i]))           for i in 1:n]
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
        crps_i     = round.(crps_p[1],     digits=4),
    )
    sort!(df_p, :y_observed, rev=true)
    CSV.write("results/walkfree/poisson/tables/country_summary_poisson.csv", df_p)
    println("  Country summary saved.")
end

println("\nPoisson section complete.")

# ============================================================================
# Section 5 – NB grid search via pmap
# ============================================================================

println("\n[Section 5] NBPopulationRatesMarg — grid search over r via pmap")
println("  r_grid = $r_grid  ($(length(r_grid)) values, $(nworkers()) workers)")

grid_results = pmap(r_grid) do r_fixed
    run_grid_r(r_fixed, y, P, D, n_burnin, ddcrp_params, priors_marg, opts_nb)
end

# pmap preserves order, but sort by r to be safe
sort!(grid_results, by = res -> res.r)

for res in grid_results
    @printf "  r=%-5.1f  K_mean=%-6.2f  WAIC=%-12.2f  ELPD-LOO=%-12.2f  CRPS=%.4f\n" res.r res.mean_K res.waic res.elpd_loo res.crps
end

println("\n[Section 5] MAP cluster membership for each r value:")
for res in grid_results
    csizes = countmap(res.z_map)
    println("\n  === NB r=$(res.r) — MAP clusters ===")
    for (cl, cnt) in sort(collect(csizes), by=x -> x[2])
        members = sort(String.(df_clean.Country[findall(==(cl), res.z_map)]))
        println("    Cluster $cl (n=$cnt): " * join(members, ", "))
    end
end

# ============================================================================
# Section 6 – Model comparison and r selection
# ============================================================================

println("\n[Section 6] Model comparison")

df_grid = DataFrame(
    model     = vcat(["NB r=$(res.r)" for res in grid_results], ["Poisson"]),
    r_fixed   = vcat([res.r           for res in grid_results], [NaN]),
    mean_K    = vcat([res.mean_K      for res in grid_results], [mean(k_post_p)]),
    median_K  = vcat([res.median_K    for res in grid_results], [median(k_post_p)]),
    mode_K    = vcat([res.mode_K      for res in grid_results], [argmax(countmap(k_post_p))]),
    waic      = vcat([res.waic        for res in grid_results], [waic_p.waic]),
    elpd_loo  = vcat([res.elpd_loo    for res in grid_results], [loo_p.elpd_loo]),
    crps      = vcat([res.crps        for res in grid_results], [crps_p[2]]),
    mal       = vcat([res.mal         for res in grid_results], [mal_p]),
    k_hat_bad = vcat([res.k_hat_bad   for res in grid_results], [sum(loo_p.k_hat .> 0.7)]),
    time_s    = vcat([res.time_s      for res in grid_results], [diag_poisson.total_time]),
)
CSV.write("results/walkfree/model_comparison.csv", df_grid)
println("  Saved to results/walkfree/model_comparison.csv")

nb_rows   = filter(r -> r.model != "Poisson", df_grid)
sorted_nb = sort(nb_rows, :waic)
println("\n  NB grid — top 10 by WAIC (lower is better):")
println("  " * rpad("r", 6) * rpad("WAIC", 12) * rpad("ELPD-LOO", 14) * rpad("CRPS", 10) * rpad("MAL", 10) * rpad("mean_K", 10) * "k̂>0.7")
println("  " * "-"^68)
for row in eachrow(sorted_nb[1:min(10, nrow(sorted_nb)), :])
    @printf "  %-6.1f %-12.2f %-14.2f %-10.4f %-10.4f %-10.2f %d\n" row.r_fixed row.waic row.elpd_loo row.crps row.mal row.mean_K row.k_hat_bad
end

poisson_row = filter(r -> r.model == "Poisson", df_grid)[1, :]
println("\n  Poisson: WAIC=$(round(poisson_row.waic, digits=2))  ELPD-LOO=$(round(poisson_row.elpd_loo, digits=2))  CRPS=$(round(poisson_row.crps, digits=4))  mean_K=$(round(poisson_row.mean_K, digits=2))")

best_idx = argmax([res.elpd_loo for res in grid_results])
best_r   = grid_results[best_idx].r
println("\n  Best r (by ELPD-LOO) = $(best_r)")
@printf "  Best ELPD-LOO = %.2f\n" grid_results[best_idx].elpd_loo
@printf "  ΔELPD-LOO (best NB - Poisson) = %.2f\n" (grid_results[best_idx].elpd_loo - poisson_row.elpd_loo)

r_vals_plot = [res.r for res in grid_results]

p_waic_r = plot(r_vals_plot, [res.waic    for res in grid_results];
                markershape=:circle, linewidth=2, color=:steelblue,
                title="WAIC vs r", xlabel="r (fixed)", ylabel="WAIC", legend=:topright)
hline!(p_waic_r, [poisson_row.waic]; color=:forestgreen, linestyle=:dash, label="Poisson")
savefig(p_waic_r, "results/walkfree/nb_grid/figures/waic_vs_r.png")

p_loo_r = plot(r_vals_plot, [res.elpd_loo for res in grid_results];
               markershape=:circle, linewidth=2, color=:steelblue,
               title="ELPD-LOO vs r", xlabel="r (fixed)", ylabel="ELPD-LOO", legend=:topright)
hline!(p_loo_r, [poisson_row.elpd_loo]; color=:forestgreen, linestyle=:dash, label="Poisson")
savefig(p_loo_r, "results/walkfree/nb_grid/figures/elpd_loo_vs_r.png")

p_crps_r = plot(r_vals_plot, [res.crps    for res in grid_results];
                markershape=:circle, linewidth=2, color=:steelblue,
                title="CRPS vs r", xlabel="r (fixed)", ylabel="CRPS", legend=:topright)
hline!(p_crps_r, [poisson_row.crps]; color=:forestgreen, linestyle=:dash, label="Poisson")
savefig(p_crps_r, "results/walkfree/nb_grid/figures/crps_vs_r.png")

p_k_r = plot(r_vals_plot, [res.mean_K    for res in grid_results];
             markershape=:circle, linewidth=2, color=:steelblue,
             title="Mean K vs r", xlabel="r (fixed)", ylabel="Mean K", legend=:topright)
hline!(p_k_r, [mean(k_post_p)]; color=:forestgreen, linestyle=:dash, label="Poisson")
savefig(p_k_r, "results/walkfree/nb_grid/figures/mean_k_vs_r.png")

p_metrics = plot(p_waic_r, p_loo_r, p_crps_r, p_k_r, layout=(2, 2), size=(1000, 700),
                 plot_title="Model comparison metrics vs r")
savefig(p_metrics, "results/walkfree/nb_grid/figures/metric_vs_r.png")

println("\nMetric vs r figures saved to results/walkfree/nb_grid/figures/")

# ============================================================================
# Section 7 – Best-r NB extended post-processing (main process)
# ============================================================================

println("\n[Section 7] Extended post-processing for best r = $(best_r)")

opts_nb_verbose = MCMCOptions(
    n_samples         = n_samples,
    verbose           = false,
    track_diagnostics = true,
    infer_params      = Dict(:r => false, :c => true, :α_ddcrp => true, :s_ddcrp => true),
    prop_sds          = Dict(:s_ddcrp => 0.3)
)

println("  Re-running best r=$(best_r) with verbose output...")
samples_best, diag_best = mcmc(
    NBPopulationRatesMarg(),
    y, P, D,
    ddcrp_params,
    priors_marg,
    ConjugateProposal();
    opts        = opts_nb_verbose,
    init_params = Dict{Symbol,Any}(:r => Float64(best_r))
)
println("  Total time: $(round(diag_best.total_time, digits=1)) s")
@save "results/walkfree/nb_best/chains/samples_best.jld2" samples_best diag_best ddcrp_params priors_marg best_r
println("  Chain saved.")

idx_b    = (n_burnin + 1):size(samples_best.c, 1)
c_post_b = samples_best.c[idx_b, :]
α_post_b = samples_best.α_ddcrp[idx_b]
s_post_b = samples_best.s_ddcrp[idx_b]
λ_post_b = samples_best.λ[idx_b, :]

println("\n--- NBPopulationRatesMarg best r=$(best_r) ($(length(idx_b)) post-burnin samples) ---")
k_post_b = calculate_n_clusters(c_post_b)
println("  K:  mean=$(round(mean(k_post_b), digits=2))  median=$(median(k_post_b))  mode=$(argmax(countmap(k_post_b)))")
println("  α:  mean=$(round(mean(α_post_b), digits=3))  95%CI=[$(round(quantile(α_post_b,0.025),digits=3)), $(round(quantile(α_post_b,0.975),digits=3))]")
println("  s:  mean=$(round(mean(s_post_b), digits=3))  95%CI=[$(round(quantile(s_post_b,0.025),digits=3)), $(round(quantile(s_post_b,0.975),digits=3))]")
println("  ESS(K)=$(round(effective_sample_size(Float64.(k_post_b)), digits=1))  ESS(α)=$(round(effective_sample_size(α_post_b), digits=1))  ESS(s)=$(round(effective_sample_size(s_post_b), digits=1))")

k_post_b, sim_b = save_standard_figures(
    c_post_b, α_post_b, s_post_b,
    samples_best.logpost[idx_b],
    "results/walkfree/nb_best/figures/nb_best",
    "NB best r=$(best_r)", :steelblue
)

z_map_b, csizes_b = print_map_clusters(c_post_b, df_clean, "NB best r=$(best_r)")
k_sorted_b = sort(collect(countmap(k_post_b)), by=x -> -x[2])
println("  K posterior (top 10):")
for (k, cnt) in k_sorted_b[1:min(10, length(k_sorted_b))]
    @printf "    K=%-4d  p=%.4f\n" k cnt/length(k_post_b)
end

eff_rates_b = λ_post_b .* Float64.(P)'
ypred_b     = ppc_from_rates(eff_rates_b, P)
ll_mat_b    = compute_ll_matrix(y, eff_rates_b)
waic_b      = compute_waic(y, eff_rates_b)
loo_b       = compute_psis_loo(ll_mat_b)
crps_b      = compute_crps(y, ypred_b)
mal_b       = compute_mal(y, ypred_b)
println("  WAIC=$(round(waic_b.waic, digits=2))  ELPD-LOO=$(round(loo_b.elpd_loo, digits=2))  CRPS=$(round(crps_b[2], digits=4))  MAL=$(round(mal_b, digits=4))")
println("  k̂ > 0.7: $(sum(loo_b.k_hat .> 0.7)) / $n obs")

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
        crps_i     = round.(crps_b[1],     digits=4),
    )
    sort!(df_b, :y_observed, rev=true)
    CSV.write("results/walkfree/nb_best/tables/country_summary_nb_best.csv", df_b)
    println("  Country table saved.")
end

# Comparison: Poisson vs NB best
p_cmp_dens = plot(title="PPC — Poisson vs NB r=$(best_r)", xlabel="log₁₀(y+1)", ylabel="Density")
density!(p_cmp_dens, vec(log10.(Float64.(ypred_p) .+ 1)); label="Poisson PP", color=:forestgreen, linewidth=1.5, alpha=0.7)
density!(p_cmp_dens, vec(log10.(Float64.(ypred_b) .+ 1)); label="NB r=$(best_r) PP", color=:steelblue, linewidth=1.5, alpha=0.7)
density!(p_cmp_dens, log10.(Float64.(y) .+ 1); label="Observed", color=:black, linewidth=2, linestyle=:dash)
savefig(p_cmp_dens, "results/walkfree/figures/ppc_comparison.png")

p_cmp_k = plot(title="K distribution — Poisson vs NB r=$(best_r)", xlabel="K", ylabel="Probability")
let cm = countmap(k_post_p); ks = sort(collect(keys(cm)))
    plot!(p_cmp_k, ks, [cm[k]/length(k_post_p) for k in ks]; label="Poisson", color=:forestgreen, markershape=:circle, linewidth=2)
end
let cm = countmap(k_post_b); ks = sort(collect(keys(cm)))
    plot!(p_cmp_k, ks, [cm[k]/length(k_post_b) for k in ks]; label="NB r=$(best_r)", color=:steelblue, markershape=:circle, linewidth=2)
end
savefig(p_cmp_k, "results/walkfree/figures/k_distribution_comparison.png")

println("\nBest-r NB section complete.")

# ============================================================================
# Section 8 – RJMCMC confirmation at best r (main process)
# ============================================================================

println("\n[Section 8] RJMCMC confirmation with best r=$(best_r)")

opts_rj = MCMCOptions(
    n_samples         = n_samples,
    verbose           = false,
    track_diagnostics = true,
    infer_params      = Dict(:r => false, :c => true, :α_ddcrp => true, :s_ddcrp => true),
    prop_sds          = Dict(:s_ddcrp => 0.3)
)

samples_rj, diag_rj = mcmc(
    NBPopulationRates(),
    y, P, D,
    ddcrp_params,
    priors_unmarg,
    PriorProposal();
    fixed_dim_proposal = NoUpdate(),
    opts               = opts_rj,
    init_params        = Dict{Symbol,Any}(:r => Float64(best_r))
)
ar_rj = acceptance_rates(diag_rj)
println("  Acceptance rates — birth: $(round(ar_rj.birth, digits=3))  death: $(round(ar_rj.death, digits=3))  fixed: $(round(ar_rj.fixed, digits=3))")
println("  Total time: $(round(diag_rj.total_time, digits=1)) s")
@save "results/walkfree/rjmcmc/chains/samples_rjmcmc.jld2" samples_rj diag_rj ddcrp_params priors_unmarg best_r
println("  Chain saved.")

idx_rj    = (n_burnin + 1):size(samples_rj.c, 1)
c_post_rj = samples_rj.c[idx_rj, :]
α_post_rj = samples_rj.α_ddcrp[idx_rj]
s_post_rj = samples_rj.s_ddcrp[idx_rj]
k_post_rj = calculate_n_clusters(c_post_rj)

println("\n--- RJMCMC best r=$(best_r) ($(length(idx_rj)) post-burnin samples) ---")
println("  K:  mean=$(round(mean(k_post_rj), digits=2))  median=$(median(k_post_rj))  mode=$(argmax(countmap(k_post_rj)))")
println("  α:  mean=$(round(mean(α_post_rj), digits=3))  95%CI=[$(round(quantile(α_post_rj,0.025),digits=3)), $(round(quantile(α_post_rj,0.975),digits=3))]")
println("  s:  mean=$(round(mean(s_post_rj), digits=3))  95%CI=[$(round(quantile(s_post_rj,0.025),digits=3)), $(round(quantile(s_post_rj,0.975),digits=3))]")
println("  ESS(K)=$(round(effective_sample_size(Float64.(k_post_rj)), digits=1))")

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

println("\nRJMCMC vs Marg-Gibbs comparison figures saved to results/walkfree/figures/")

# ============================================================================
# Section 9 – Best-per-metric RJMCMC (parallel via pmap)
# ============================================================================

println("\n[Section 9] Best-per-metric RJMCMC confirmation")

@everywhere function run_best_rjmcmc(r_fixed, metric_label, y, P, D, n_burnin,
                                     ddcrp_params, priors_unmarg, opts_rj)
    r_tag   = replace(@sprintf("%.1f", r_fixed), "." => "p")
    out_dir = "results/walkfree/best_models/$(metric_label)_r$(r_tag)"
    mkpath("$(out_dir)/figures")
    mkpath("$(out_dir)/chains")

    t_start = time()
    samples_rj, diag_rj = mcmc(
        NBPopulationRates(),
        y, P, D,
        ddcrp_params, priors_unmarg,
        PriorProposal();
        fixed_dim_proposal = NoUpdate(),
        opts               = opts_rj,
        init_params        = Dict{Symbol,Any}(:r => Float64(r_fixed))
    )
    t_elapsed = time() - t_start

    @save "$(out_dir)/chains/samples_rjmcmc.jld2" samples_rj diag_rj r_fixed metric_label

    idx_rj    = (n_burnin + 1):size(samples_rj.c, 1)
    c_post_rj = samples_rj.c[idx_rj, :]
    α_post_rj = samples_rj.α_ddcrp[idx_rj]
    s_post_rj = samples_rj.s_ddcrp[idx_rj]
    k_post_rj = calculate_n_clusters(c_post_rj)

    save_standard_figures(c_post_rj, α_post_rj, s_post_rj,
        samples_rj.logpost[idx_rj],
        "$(out_dir)/figures/rjmcmc",
        "RJMCMC r=$(r_fixed) [$(metric_label)]", :darkorange)

    ar_rj = acceptance_rates(diag_rj)
    return (
        r            = r_fixed,
        metric_label = metric_label,
        mean_K       = mean(k_post_rj),
        median_K     = median(k_post_rj),
        mode_K       = argmax(countmap(k_post_rj)),
        mean_α       = mean(α_post_rj),
        mean_s       = mean(s_post_rj),
        ess_K        = effective_sample_size(Float64.(k_post_rj)),
        ar_birth     = ar_rj.birth,
        ar_death     = ar_rj.death,
        time_s       = t_elapsed,
    )
end

# Identify best r for each metric
best_by_waic = grid_results[argmin([res.waic     for res in grid_results])].r
best_by_crps = grid_results[argmin([res.crps     for res in grid_results])].r
best_by_elpd = grid_results[argmax([res.elpd_loo for res in grid_results])].r
best_by_mal  = grid_results[argmin([res.mal      for res in grid_results])].r

println("  Best r by WAIC     = $(best_by_waic)")
println("  Best r by CRPS     = $(best_by_crps)")
println("  Best r by ELPD-LOO = $(best_by_elpd)")
println("  Best r by MAL      = $(best_by_mal)")

# Collect (r => metrics it wins) mapping, then run each unique r once
metric_winners = [
    (best_by_waic, "best_waic"),
    (best_by_crps, "best_crps"),
    (best_by_elpd, "best_elpd"),
    (best_by_mal,  "best_mal"),
]
unique_pairs = Dict{Float64, Vector{String}}()
for (r, lbl) in metric_winners
    push!(get!(unique_pairs, r, String[]), lbl)
end
jobs = sort([(r, join(lbls, "_AND_")) for (r, lbls) in unique_pairs], by = x -> x[1])
println("  Unique r values to run: $([j[1] for j in jobs])")

mkpath("results/walkfree/best_models")

opts_best = MCMCOptions(
    n_samples         = n_samples,
    verbose           = false,
    track_diagnostics = true,
    infer_params      = Dict(:r => false, :c => true, :α_ddcrp => true, :s_ddcrp => true),
    prop_sds          = Dict(:s_ddcrp => 0.3)
)

println("  Running $(length(jobs)) RJMCMC chain(s) in parallel via pmap...")
best_results = pmap(jobs) do (r_fixed, label)
    run_best_rjmcmc(r_fixed, label, y, P, D, n_burnin,
                    ddcrp_params, priors_unmarg, opts_best)
end
sort!(best_results, by = res -> res.r)

# ── Print results ─────────────────────────────────────────────────────────────
println("\n=== Best-per-metric RJMCMC results ===")
hdr = rpad("Metric label", 32) * rpad("r", 7) * rpad("mean_K", 9) *
      rpad("mode_K", 9) * rpad("mean_α", 10) * rpad("mean_s", 10) *
      rpad("ESS(K)", 9) * rpad("ar_birth", 10) * rpad("ar_death", 10) * "time(s)"
println(hdr)
println("-"^112)
for res in best_results
    @printf "%-32s %-7.1f %-9.2f %-9d %-10.3f %-10.3f %-9.1f %-10.3f %-10.3f %.1f\n" \
        res.metric_label res.r res.mean_K res.mode_K res.mean_α res.mean_s \
        res.ess_K res.ar_birth res.ar_death res.time_s
end

# Marg-Gibbs grid reference for the same r values
best_r_set = Set([res.r for res in best_results])
ref_rows   = filter(res -> res.r in best_r_set, grid_results)
println("\n=== Marg-Gibbs reference (from grid search) ===")
println(rpad("r", 7) * rpad("mean_K", 9) * rpad("WAIC", 13) *
        rpad("ELPD-LOO", 14) * rpad("CRPS", 10) * "MAL")
println("-"^55)
for res in ref_rows
    @printf "%-7.1f %-9.2f %-13.2f %-14.2f %-10.4f %.4f\n" \
        res.r res.mean_K res.waic res.elpd_loo res.crps res.mal
end

# Save summary CSV
df_best = DataFrame(
    metric_label = [res.metric_label for res in best_results],
    r            = [res.r            for res in best_results],
    mean_K       = [res.mean_K       for res in best_results],
    mode_K       = [res.mode_K       for res in best_results],
    mean_α       = [res.mean_α       for res in best_results],
    mean_s       = [res.mean_s       for res in best_results],
    ess_K        = [res.ess_K        for res in best_results],
    ar_birth     = [res.ar_birth     for res in best_results],
    ar_death     = [res.ar_death     for res in best_results],
    time_s       = [res.time_s       for res in best_results],
)
CSV.write("results/walkfree/best_models/rjmcmc_best_per_metric.csv", df_best)
println("\n  Summary saved to results/walkfree/best_models/rjmcmc_best_per_metric.csv")
println("\nBest-per-metric RJMCMC section complete.")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^60)
println("Analysis complete.")
println("  Best r (ELPD-LOO): $(best_r)")
@printf "  ΔELPD-LOO (best NB - Poisson): %.2f\n" (grid_results[best_idx].elpd_loo - poisson_row.elpd_loo)
@printf "  Poisson mean K: %.2f,  NB best mean K: %.2f\n" mean(k_post_p) mean(k_post_b)
println("  Marg-Gibbs K (r=$(best_r)):  mean=$(round(mean(k_post_b), digits=2))")
println("  RJMCMC K (r=$(best_r)):       mean=$(round(mean(k_post_rj), digits=2))")
println("  k̂ > 0.7 obs: Poisson=$(sum(loo_p.k_hat .> 0.7))  NB best=$(sum(loo_b.k_hat .> 0.7))")
println("="^60)
println("\nKey output files:")
println("  results/walkfree/model_comparison.csv")
println("  results/walkfree/nb_grid/figures/metric_vs_r.png")
println("  results/walkfree/figures/k_distribution_comparison.png")
println("  results/walkfree/figures/k_distribution_marg_vs_rjmcmc.png")
println("  results/walkfree/nb_best/tables/country_summary_nb_best.csv")
