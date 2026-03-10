# ============================================================================
# Walk Free Foundation Global Slavery Index - ddCRP Analysis (chains only)
# ============================================================================
#
# Runs all MCMC samplers and saves chains + model-comparison CSV.
# Produces NO figures or tables — run walk_free_plots.jl afterwards.
#
# Structure:
#   1. Poisson model — establishes baseline (tends to over-estimate K)
#   2. NB grid search over r ∈ {0.5, 1.0, ..., 20.0} with r fixed (parallel)
#   3. Model comparison: ELPD-LOO → model_comparison.csv
#   4. Best-r NB post-processing (extended diagnostics)
#   5. RJMCMC confirmation at best r
#   6. RJMCMC at best ELPD-LOO r
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
    using DDCRP
    using CSV, DataFrames, Distances
    using Statistics, StatsBase, LinearAlgebra
    using Random
    using Printf
    using XLSX
    using JLD2
    using Distributions
    using SpecialFunctions
end

Random.seed!(123)

# ============================================================================
# Section 0 – Output directories
# ============================================================================

mkpath("results/walkfree")
mkpath("results/walkfree/chains")
mkpath("results/walkfree/poisson/chains")
mkpath("results/walkfree/poisson/tables")
mkpath("results/walkfree/nb_grid/chains")
mkpath("results/walkfree/nb_best/chains")
mkpath("results/walkfree/nb_best/tables")
mkpath("results/walkfree/rjmcmc/chains")
mkpath("results/walkfree/rjmcmc_elpd/chains")

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
    n_samples = 125_000
    n_burnin  = 25_000
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

"""
Run one r-value of the NB grid search. Called via pmap.
Saves chain; returns a metrics NamedTuple.
"""
function run_grid_r(r_fixed, y, P, D, n_burnin, ddcrp_params, priors_marg, opts_nb)
    r_tag = replace(@sprintf("%.1f", r_fixed), "." => "p")
    r_dir = "results/walkfree/nb_grid/r$(r_tag)"
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

    z_map_r = point_estimate_clustering(c_post_r; method=:MAP)

    eff_rates_r = λ_post_r .* Float64.(P)'
    ll_mat_r    = compute_ll_matrix(y, eff_rates_r)
    waic_r      = compute_waic(y, eff_rates_r)
    loo_r       = compute_psis_loo(ll_mat_r)

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
        time_s    = t_elapsed,
        z_map     = z_map_r,
    )
end

end # @everywhere

# Main-process-only helpers

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

z_map_p, csizes_p = print_map_clusters(c_post_p, df_clean, "PoissonPopulationRatesMarg")
k_sorted_p = sort(collect(countmap(k_post_p)), by=x -> -x[2])
println("  K posterior (top 10):")
for (k, cnt) in k_sorted_p[1:min(10, length(k_sorted_p))]
    @printf "    K=%-4d  p=%.4f\n" k cnt/length(k_post_p)
end

println("\n[Poisson] Computing predictive metrics...")
eff_rates_p = poisson_rate_samples(c_post_p, y, P, priors_poisson)
ll_mat_p    = compute_ll_matrix(y, eff_rates_p)
waic_p      = compute_waic(y, eff_rates_p)
loo_p       = compute_psis_loo(ll_mat_p)
println("  WAIC=$(round(waic_p.waic, digits=2))  ELPD-LOO=$(round(loo_p.elpd_loo, digits=2))")
println("  k̂ > 0.7 for $(sum(loo_p.k_hat .> 0.7)) / $n observations")

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
    @printf "  r=%-5.1f  K_mean=%-6.2f  WAIC=%-12.2f  ELPD-LOO=%-12.2f\n" res.r res.mean_K res.waic res.elpd_loo
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
    k_hat_bad = vcat([res.k_hat_bad   for res in grid_results], [sum(loo_p.k_hat .> 0.7)]),
    time_s    = vcat([res.time_s      for res in grid_results], [diag_poisson.total_time]),
)
CSV.write("results/walkfree/model_comparison.csv", df_grid)
println("  Saved to results/walkfree/model_comparison.csv")

nb_rows   = filter(r -> r.model != "Poisson", df_grid)
sorted_nb = sort(nb_rows, :waic)
println("\n  NB grid — top 10 by WAIC (lower is better):")
println("  " * rpad("r", 6) * rpad("WAIC", 12) * rpad("ELPD-LOO", 14) * rpad("mean_K", 10) * "k̂>0.7")
println("  " * "-"^48)
for row in eachrow(sorted_nb[1:min(10, nrow(sorted_nb)), :])
    @printf "  %-6.1f %-12.2f %-14.2f %-10.2f %d\n" row.r_fixed row.waic row.elpd_loo row.mean_K row.k_hat_bad
end

poisson_row = filter(r -> r.model == "Poisson", df_grid)[1, :]
println("\n  Poisson: WAIC=$(round(poisson_row.waic, digits=2))  ELPD-LOO=$(round(poisson_row.elpd_loo, digits=2))  mean_K=$(round(poisson_row.mean_K, digits=2))")

best_idx = argmax([res.elpd_loo for res in grid_results])
best_r   = grid_results[best_idx].r
println("\n  Best r (by ELPD-LOO) = $(best_r)")
@printf "  Best ELPD-LOO = %.2f\n" grid_results[best_idx].elpd_loo
@printf "  ΔELPD-LOO (best NB - Poisson) = %.2f\n" (grid_results[best_idx].elpd_loo - poisson_row.elpd_loo)

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

println("  Re-running best r=$(best_r)...")
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

z_map_b, csizes_b = print_map_clusters(c_post_b, df_clean, "NB best r=$(best_r)")
k_sorted_b = sort(collect(countmap(k_post_b)), by=x -> -x[2])
println("  K posterior (top 10):")
for (k, cnt) in k_sorted_b[1:min(10, length(k_sorted_b))]
    @printf "    K=%-4d  p=%.4f\n" k cnt/length(k_post_b)
end

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
k_post_rj = calculate_n_clusters(samples_rj.c[idx_rj, :])
println("\n--- RJMCMC best r=$(best_r) ($(length(idx_rj)) post-burnin samples) ---")
println("  K:  mean=$(round(mean(k_post_rj), digits=2))  median=$(median(k_post_rj))  mode=$(argmax(countmap(k_post_rj)))")
println("  ESS(K)=$(round(effective_sample_size(Float64.(k_post_rj)), digits=1))")

# ============================================================================
# Section 9 – RJMCMC best ELPD-LOO
# ============================================================================

println("\n[Section 9] RJMCMC best ELPD-LOO")

best_by_elpd = grid_results[argmax([res.elpd_loo for res in grid_results])].r
println("  Best r by ELPD-LOO = $(best_by_elpd)")

opts_best = MCMCOptions(
    n_samples         = n_samples,
    verbose           = false,
    track_diagnostics = true,
    infer_params      = Dict(:r => false, :c => true, :α_ddcrp => true, :s_ddcrp => true),
    prop_sds          = Dict(:s_ddcrp => 0.3)
)

Random.seed!(2025)
t_start_best = time()
samples_best_rj, diag_best_rj = mcmc(
    NBPopulationRates(),
    y, P, D,
    ddcrp_params, priors_unmarg,
    PriorProposal();
    fixed_dim_proposal = NoUpdate(),
    opts               = opts_best,
    init_params        = Dict{Symbol,Any}(:r => Float64(best_by_elpd))
)
t_elapsed_best = time() - t_start_best
println("  Total time: $(round(t_elapsed_best, digits=1)) s")

@save "results/walkfree/rjmcmc_elpd/chains/samples_rjmcmc.jld2" samples_best_rj diag_best_rj best_by_elpd
println("  Chain saved.")

ar_best = acceptance_rates(diag_best_rj)
idx_best    = (n_burnin + 1):size(samples_best_rj.c, 1)
k_post_best = calculate_n_clusters(samples_best_rj.c[idx_best, :])

println("\n=== Best ELPD-LOO RJMCMC results (r=$(best_by_elpd)) ===")
println("  mean_K   = $(round(mean(k_post_best),   digits=2))")
println("  mode_K   = $(argmax(countmap(k_post_best)))")
println("  mean_α   = $(round(mean(samples_best_rj.α_ddcrp[idx_best]), digits=3))")
println("  mean_s   = $(round(mean(samples_best_rj.s_ddcrp[idx_best]), digits=3))")
println("  ESS(K)   = $(round(effective_sample_size(Float64.(k_post_best)), digits=1))")
println("  ar_birth = $(round(ar_best.birth, digits=3))")
println("  ar_death = $(round(ar_best.death, digits=3))")
@printf "  time     = %.1f s\n" t_elapsed_best

df_best_summary = DataFrame(
    r        = [best_by_elpd],
    mean_K   = [mean(k_post_best)],
    mode_K   = [argmax(countmap(k_post_best))],
    mean_α   = [mean(samples_best_rj.α_ddcrp[idx_best])],
    mean_s   = [mean(samples_best_rj.s_ddcrp[idx_best])],
    ess_K    = [effective_sample_size(Float64.(k_post_best))],
    ar_birth = [ar_best.birth],
    ar_death = [ar_best.death],
    time_s   = [t_elapsed_best],
)
CSV.write("results/walkfree/rjmcmc_elpd/rjmcmc_best_elpd.csv", df_best_summary)
println("\n  Summary saved to results/walkfree/rjmcmc_elpd/rjmcmc_best_elpd.csv")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^60)
println("Analysis complete. Run walk_free_plots.jl to generate figures.")
println("  Best r (ELPD-LOO): $(best_r)")
@printf "  ΔELPD-LOO (best NB - Poisson): %.2f\n" (grid_results[best_idx].elpd_loo - poisson_row.elpd_loo)
@printf "  Poisson mean K: %.2f,  NB best mean K: %.2f\n" mean(k_post_p) mean(k_post_b)
println("="^60)
println("\nKey output files:")
println("  results/walkfree/model_comparison.csv")
println("  results/walkfree/poisson/chains/samples_poisson.jld2")
println("  results/walkfree/nb_best/chains/samples_best.jld2")
println("  results/walkfree/rjmcmc/chains/samples_rjmcmc.jld2")
println("  results/walkfree/rjmcmc_elpd/chains/samples_rjmcmc.jld2")
