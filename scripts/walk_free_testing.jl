# ============================================================================
# Walk Free Foundation Global Slavery Index — Scenario Testing (distributed)
# ============================================================================
#
# Runs 5 scenarios on the Walk Free GSI data using only
# NBPopulationRatesMarg (Gibbs / marginalised γ_k) + ConjugateProposal().
#
# Scenario 1: Fixed cluster assignments (no DDCRP update)
#             Sub-runs: one_cluster, twenty_clusters, all_singletons
# Scenario 2: Window distance, threshold ∈ {0.1, 0.5, 1.0, …, 5.0}
# Scenario 3: KNN distance, K ∈ {1, 2, …, 10}
# Scenario 4: Cosine distance + exponential decay (α, s inferred)
# Scenario 5: Manhattan (L1) distance + exponential decay (α, s inferred)
#
# CLI options:
#   --test          reduced iteration counts for a quick smoke-test
#   --nprocs N      number of local worker processes (default: Sys.CPU_THREADS ÷ 2)
#   --slurm         use SlurmClusterManager instead of local workers
#
# Results saved to results/walkfree_testing/<scenario>/
# ============================================================================

using Distributed

# ── CLI args (must be parsed before addprocs) ─────────────────────────────────
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
    using ClusterManagers
    addprocs(SlurmManager())
    println("Added $(nworkers()) SLURM workers")
elseif N_PROCS >= 1
    addprocs(N_PROCS)
    println("Added $(nworkers()) local workers")
end

# In parallel mode suppress per-iteration MCMC output to avoid interleaving
const VERBOSE = nworkers() == 1

# ── Packages on all processes ─────────────────────────────────────────────────
@everywhere begin
    ENV["GKSwstype"] = "100"   # headless Plots.jl (required on worker nodes)
    using DDCRP
    using CSV, DataFrames, Distances
    using Statistics, StatsBase, LinearAlgebra
    using Plots, StatsPlots
    using Random
    using Printf
    using JLD2
    using Distributions
end

# ── Shared constant on all processes ─────────────────────────────────────────
@everywhere const PRIORS = NBPopulationRatesMargPriors(1.0, 0.1, 1.0, 0.1)

# ── Helper functions and post-processor (defined on all processes) ────────────
@everywhere begin

identity_decay(w; kwargs...) = w

function window_distance_matrix(D::AbstractMatrix, threshold::Real)
    n = size(D, 1)
    W = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        i == j && continue
        W[i, j] = D[i, j] < threshold ? 1.0 : 0.0
    end
    return W
end

function knn_distance_matrix(D::AbstractMatrix, K::Int)
    n = size(D, 1)
    W = zeros(Float64, n, n)
    for i in 1:n
        js    = [j for j in 1:n if j != i]
        order = sortperm([D[i, j] for j in js])
        for j in js[order[1:min(K, length(js))]]
            W[i, j] = 1.0
        end
    end
    return W
end

function posterior_predictive_samples(λ_post::Matrix{Float64}, P::Vector{Int})
    n_iter, n_obs = size(λ_post)
    y_pred = zeros(Int, n_iter, n_obs)
    for s in 1:n_iter, i in 1:n_obs
        y_pred[s, i] = rand(Poisson(P[i] * λ_post[s, i]))
    end
    return y_pred
end

"""
    posterior_predictive_ignore_lambda(c_post, λ_post, r_post, P, γ_a, γ_b)

Alternative PPD that re-samples γ_k from its conjugate posterior given stored λ's,
then draws fresh λ_i ~ Gamma(r, γ_k/r) before simulating y.

This breaks the tight λ_i ↔ y_i coupling and reveals clustering structure in
the predictive distribution.
"""
function posterior_predictive_ignore_lambda(
    c_post  :: Matrix{Int},
    λ_post  :: Matrix{Float64},
    r_post  :: Vector{Float64},
    P       :: Vector{Int},
    γ_a     :: Float64,
    γ_b     :: Float64
)
    n_iter, n_obs = size(λ_post)
    y_pred = zeros(Int, n_iter, n_obs)
    for s in 1:n_iter
        r      = r_post[s]
        labels = c_to_z(c_post[s, :], n_obs)
        for k in unique(labels)
            idx_k = findall(==(k), labels)
            n_k   = length(idx_k)
            Λ_k   = sum(λ_post[s, i] for i in idx_k)
            γ_k   = rand(InverseGamma(γ_a + n_k * r, γ_b + r * Λ_k))
            for i in idx_k
                λ_new        = rand(Gamma(r, γ_k / r))
                y_pred[s, i] = rand(Poisson(P[i] * λ_new))
            end
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

"""
    run_postprocess(samples, diag, label, outdir, n_burnin, y, P, df_clean,
                    pc1, pc2, pct1, pct2; has_s, c_fixed, color, priors)

Full post-processing for one MCMC run: statistics, figures, CSVs.
Returns a slim NamedTuple of posterior summaries (no large λ/sim arrays).
"""
function run_postprocess(
    samples, diag, label, outdir, n_burnin, y, P, df_clean,
    pc1, pc2, pct1, pct2;
    has_s::Bool   = true,
    c_fixed::Bool = false,
    color         = :steelblue,
    priors        = PRIORS
)
    mkpath(outdir)
    mkpath(joinpath(outdir, "figures"))

    idx    = (n_burnin + 1):size(samples.c, 1)
    c_post = samples.c[idx, :]
    r_post = samples.r[idx]
    λ_post = samples.λ[idx, :]
    α_post = samples.α_ddcrp[idx]
    s_post = samples.s_ddcrp[idx]
    k_post = c_fixed ? fill(length(unique(c_post[1, :])), length(idx)) :
                       calculate_n_clusters(c_post)

    ess_k = c_fixed ? NaN : effective_sample_size(Float64.(k_post))
    ess_r = effective_sample_size(r_post)
    ess_α = effective_sample_size(α_post)
    ess_s = has_s ? effective_sample_size(s_post) : NaN

    println("\n--- $label (post burn-in = $(length(idx)) samples) ---")
    if !c_fixed
        println("  K:  mean=$(round(mean(k_post), digits=2))  " *
                "median=$(median(k_post))  mode=$(argmax(countmap(k_post)))")
    else
        println("  K:  FIXED = $(k_post[1])")
    end
    println("  r:  mean=$(round(mean(r_post), digits=3))  " *
            "95%CI=[$(round(quantile(r_post,0.025),digits=3)), " *
            "$(round(quantile(r_post,0.975),digits=3))]")
    println("  α:  mean=$(round(mean(α_post), digits=3))  " *
            "95%CI=[$(round(quantile(α_post,0.025),digits=3)), " *
            "$(round(quantile(α_post,0.975),digits=3))]")
    if has_s
        println("  s:  mean=$(round(mean(s_post), digits=3))  " *
                "95%CI=[$(round(quantile(s_post,0.025),digits=3)), " *
                "$(round(quantile(s_post,0.975),digits=3))]")
    end
    if !c_fixed
        @printf "  ESS(K)=%.1f  ESS(r)=%.1f  ESS(α)=%.1f" ess_k ess_r ess_α
        has_s && @printf "  ESS(s)=%.1f" ess_s
        println()
    end

    sim = compute_similarity_matrix(c_post)

    # ---- Cluster membership ----
    z_map  = point_estimate_clustering(c_post; method=:MAP)
    K_mode = c_fixed ? k_post[1] : argmax(countmap(k_post))
    csizes = countmap(z_map)
    n_singletons = count(==(1), values(csizes))
    println("  MAP cluster composition (K=$(length(csizes)), singletons=$n_singletons):")
    for (cl, cnt) in sort(collect(csizes), by=x -> x[2], rev=true)
        members = sort(String.(df_clean.Country[findall(==(cl), z_map)]))
        tag = cnt == 1 ? " [singleton]" : ""
        println("    Cluster $cl (n=$cnt)$tag: $(join(members, ", "))")
    end

    membership_df = DataFrame(
        country     = String.(df_clean.Country),
        cluster_map = z_map,
        K_mode      = K_mode,
    )
    CSV.write(joinpath(outdir, "cluster_membership.csv"), membership_df)

    # ---- Summary metrics CSV ----
    summary_df = DataFrame(
        label        = label,
        mean_K       = c_fixed ? k_post[1] : round(mean(k_post), digits=3),
        median_K     = c_fixed ? k_post[1] : median(k_post),
        mode_K       = K_mode,
        mean_r       = round(mean(r_post),   digits=4),
        r_ci_lo      = round(quantile(r_post, 0.025), digits=4),
        r_ci_hi      = round(quantile(r_post, 0.975), digits=4),
        mean_alpha   = round(mean(α_post),   digits=4),
        alpha_ci_lo  = round(quantile(α_post, 0.025), digits=4),
        alpha_ci_hi  = round(quantile(α_post, 0.975), digits=4),
        mean_s       = has_s ? round(mean(s_post), digits=4) : NaN,
        s_ci_lo      = has_s ? round(quantile(s_post, 0.025), digits=4) : NaN,
        s_ci_hi      = has_s ? round(quantile(s_post, 0.975), digits=4) : NaN,
        ess_K        = c_fixed ? NaN : round(ess_k, digits=1),
        ess_r        = round(ess_r, digits=1),
        ess_alpha    = round(ess_α, digits=1),
        ess_s        = has_s ? round(ess_s, digits=1) : NaN,
        total_time_s = round(diag.total_time, digits=1),
    )
    CSV.write(joinpath(outdir, "summary_metrics.csv"), summary_df)

    n_obs     = length(y)
    countries = String.(df_clean.Country)
    fdir      = joinpath(outdir, "figures")

    # ---- K trace ----
    p_kt = plot(title="K trace — $label", xlabel="Iteration", ylabel="K", legend=false)
    plot!(p_kt, k_post; color=color, alpha=0.7, linewidth=0.8)
    savefig(p_kt, joinpath(fdir, "k_trace.png"))

    # ---- K distribution ----
    if !c_fixed
        p_kd = plot(title="Posterior K — $label", xlabel="K", ylabel="Probability", legend=false)
        cm_k = countmap(k_post)
        ks   = sort(collect(keys(cm_k)))
        ps   = [cm_k[k] / length(k_post) for k in ks]
        plot!(p_kd, ks, ps; color=color, markershape=:circle, linewidth=2)
        savefig(p_kd, joinpath(fdir, "k_distribution.png"))
    end

    # ---- r trace and density ----
    p_rt = plot(title="r trace — $label", xlabel="Iteration", ylabel="r", legend=false)
    plot!(p_rt, r_post; color=color, alpha=0.7, linewidth=0.8)
    savefig(p_rt, joinpath(fdir, "r_trace.png"))

    p_rd = plot(title="Posterior r — $label", xlabel="r", ylabel="Density", legend=false)
    density!(p_rd, r_post; color=color, linewidth=2)
    savefig(p_rd, joinpath(fdir, "r_density.png"))

    # ---- α density ----
    p_αd = plot(title="Posterior α — $label", xlabel="α", ylabel="Density", legend=false)
    density!(p_αd, α_post; color=color, linewidth=2)
    savefig(p_αd, joinpath(fdir, "alpha_density.png"))

    # ---- s density (if inferred) ----
    if has_s
        p_sd = plot(title="Posterior s — $label", xlabel="s", ylabel="Density", legend=false)
        density!(p_sd, s_post; color=color, linewidth=2)
        savefig(p_sd, joinpath(fdir, "scale_density.png"))
    end

    # ---- log-posterior trace ----
    p_lp = plot(title="Log-posterior trace — $label", xlabel="Iteration",
                ylabel="Log-posterior", legend=false)
    plot!(p_lp, samples.logpost[idx]; color=color, alpha=0.7, linewidth=0.8)
    savefig(p_lp, joinpath(fdir, "logpost_trace.png"))

    # ---- Co-clustering heatmap ----
    p_sim = heatmap(sim, title="Co-clustering — $label",
                    xlabel="Country", ylabel="Country",
                    color=:viridis, colorbar_title="Pr(same cluster)",
                    aspect_ratio=:equal)
    savefig(p_sim, joinpath(fdir, "coclustering.png"))

    # ---- PPD ----
    λ_f   = Float64.(λ_post)
    ypred = posterior_predictive_samples(λ_f, P)

    p_ppc = plot(title="PPC — $label", xlabel="log₁₀(people + 1)", ylabel="Density")
    density!(p_ppc, vec(log10.(Float64.(ypred) .+ 1));
             label="PP", color=color, linewidth=1.5, alpha=0.7)
    density!(p_ppc, log10.(Float64.(y) .+ 1);
             label="Observed", color=:black, linewidth=2, linestyle=:dash)
    savefig(p_ppc, joinpath(fdir, "ppc_density.png"))

    p_ppc_sc = ppc_scatter_panel(ypred, y, color, "PP mean vs observed — $label")
    savefig(p_ppc_sc, joinpath(fdir, "ppc_scatter.png"))

    # ---- Per-country PPD CSV ----
    yp_f       = Float64.(ypred)
    country_df = DataFrame(
        country    = countries,
        population = P,
        y_observed = y,
        rate_mean  = round.([mean(λ_f[:, i])              for i in 1:n_obs], digits=6),
        rate_ci_lo = round.([quantile(λ_f[:, i], 0.025)   for i in 1:n_obs], digits=6),
        rate_ci_hi = round.([quantile(λ_f[:, i], 0.975)   for i in 1:n_obs], digits=6),
        pp_mean    = round.([mean(yp_f[:, i])              for i in 1:n_obs], digits=1),
        pp_median  = round.([median(yp_f[:, i])            for i in 1:n_obs], digits=1),
        pp_ci_lo   = round.([quantile(yp_f[:, i], 0.025)  for i in 1:n_obs], digits=1),
        pp_ci_hi   = round.([quantile(yp_f[:, i], 0.975)  for i in 1:n_obs], digits=1),
    )
    sort!(country_df, :y_observed, rev=true)
    CSV.write(joinpath(outdir, "ppd_countries.csv"), country_df)

    # ---- Per-country PPD density plots (4×3 grids) ----
    n_per_page = 12
    n_pages    = ceil(Int, n_obs / n_per_page)
    mkpath(joinpath(fdir, "ppc_countries"))
    for page in 1:n_pages
        i_start = (page - 1) * n_per_page + 1
        i_end   = min(page * n_per_page, n_obs)
        panels  = Any[]
        for i in i_start:i_end
            yobs = Float64(y[i])
            p_i  = plot(title=countries[i], titlefontsize=6,
                        xlabel="", ylabel="", legend=false,
                        framestyle=:box, tickfontsize=5)
            density!(p_i, yp_f[:, i]; color=color, linewidth=1.2, alpha=0.8)
            vline!(p_i, [yobs]; color=:black, linewidth=1.5, linestyle=:dash)
            push!(panels, p_i)
        end
        while length(panels) < n_per_page
            push!(panels, plot(axis=false, grid=false, framestyle=:none))
        end
        p_grid = plot(panels..., layout=(4, 3), size=(1000, 1100),
                      plot_title=@sprintf("PPD by country (page %d/%d) — %s  [observed = dashed]",
                                          page, n_pages, label),
                      plot_titlefontsize=7)
        savefig(p_grid, joinpath(fdir, "ppc_countries",
                                 @sprintf("ppc_countries_%03d.png", page)))
    end
    println("  Per-country PPD: $n_pages page(s) → $(joinpath(fdir, "ppc_countries"))/")

    # ---- PPD ignoring stored λ (re-sample γ_k → λ_new → y) ----
    ig_dir = joinpath(fdir, "ppd_ignore_lambda")
    mkpath(ig_dir)
    mkpath(joinpath(ig_dir, "ppc_countries"))

    ypred_ig = posterior_predictive_ignore_lambda(
        c_post, λ_f, r_post, P,
        Float64(priors.γ_a), Float64(priors.γ_b)
    )
    yig_f = Float64.(ypred_ig)

    p_ig = plot(title="PPC (ignore λ) — $label",
                xlabel="log₁₀(people + 1)", ylabel="Density")
    density!(p_ig, vec(log10.(yig_f .+ 1));
             label="PP (ignore λ)", color=color, linewidth=1.5, alpha=0.7)
    density!(p_ig, log10.(Float64.(y) .+ 1);
             label="Observed", color=:black, linewidth=2, linestyle=:dash)
    savefig(p_ig, joinpath(ig_dir, "ppc_density.png"))

    p_ig_sc = ppc_scatter_panel(ypred_ig, y, color, "PP (ignore λ) mean vs observed — $label")
    savefig(p_ig_sc, joinpath(ig_dir, "ppc_scatter.png"))

    ig_df = DataFrame(
        country    = countries,
        population = P,
        y_observed = y,
        pp_mean    = round.([mean(yig_f[:, i])             for i in 1:n_obs], digits=1),
        pp_median  = round.([median(yig_f[:, i])           for i in 1:n_obs], digits=1),
        pp_ci_lo   = round.([quantile(yig_f[:, i], 0.025)  for i in 1:n_obs], digits=1),
        pp_ci_hi   = round.([quantile(yig_f[:, i], 0.975)  for i in 1:n_obs], digits=1),
    )
    sort!(ig_df, :y_observed, rev=true)
    CSV.write(joinpath(outdir, "ppd_ignore_lambda_countries.csv"), ig_df)

    n_pages_ig = ceil(Int, n_obs / 12)
    for page in 1:n_pages_ig
        i_start = (page - 1) * 12 + 1
        i_end   = min(page * 12, n_obs)
        panels  = Any[]
        for i in i_start:i_end
            yobs = Float64(y[i])
            p_i  = plot(title=countries[i], titlefontsize=6,
                        xlabel="", ylabel="", legend=false,
                        framestyle=:box, tickfontsize=5)
            density!(p_i, yig_f[:, i]; color=color, linewidth=1.2, alpha=0.8)
            vline!(p_i, [yobs]; color=:black, linewidth=1.5, linestyle=:dash)
            push!(panels, p_i)
        end
        while length(panels) < 12
            push!(panels, plot(axis=false, grid=false, framestyle=:none))
        end
        p_grid = plot(panels..., layout=(4, 3), size=(1000, 1100),
                      plot_title=@sprintf("PPD (ignore λ) page %d/%d — %s  [observed = dashed]",
                                          page, n_pages_ig, label),
                      plot_titlefontsize=7)
        savefig(p_grid, joinpath(ig_dir, "ppc_countries",
                                 @sprintf("ppc_countries_%03d.png", page)))
    end
    println("  PPD (ignore λ): $n_pages_ig page(s) → $ig_dir/")

    # ---- PC1 vs PC2 cluster plot (skip if all singletons) ----
    keep_idx = findall(i -> csizes[z_map[i]] > 1, 1:n_obs)
    if length(keep_idx) >= 2
        unique_cl = sort(unique(z_map[keep_idx]))
        remap     = Dict(c => k for (k, c) in enumerate(unique_cl))
        z_ns      = [remap[z_map[i]] for i in keep_idx]
        K_ns      = length(unique_cl)
        p_pca = plot(
            xlabel  = "PC1 ($pct1% var explained)",
            ylabel  = "PC2 ($pct2% var explained)",
            title   = "MAP clusters — $label",
            legend  = :outerright,
            size    = (800, 500)
        )
        for k in 1:K_ns
            ji = findall(==(k), z_ns)
            scatter!(p_pca,
                pc1[keep_idx[ji]], pc2[keep_idx[ji]];
                label="Cluster $k", markersize=5, markerstrokewidth=0.4)
        end
        savefig(p_pca, joinpath(fdir, "cluster_pc1_pc2.png"))
    end

    return (
        k       = k_post,
        r       = r_post,
        α       = α_post,
        s       = s_post,
        ess_k   = ess_k,
        ess_r   = ess_r,
        ess_α   = ess_α,
        ess_s   = ess_s,
        label   = label,
        has_s   = has_s,
        c_fixed = c_fixed,
    )
end

end  # @everywhere begin (helpers + run_postprocess)

# ── Task runner (on all processes) ───────────────────────────────────────────
@everywhere function run_scenario_task(task)
    println("\n[$(task.label)]  starting MCMC on pid=$(myid())...")
    mkpath(joinpath(task.outdir, "chains"))

    samples, diag = mcmc(
        NBPopulationRatesMarg(),
        task.y, task.P, task.D,
        task.ddcrp_params, task.priors,
        ConjugateProposal();
        opts   = task.opts,
        init_c = task.init_c
    )
    println("  [$(task.label)]  MCMC done ($(round(diag.total_time, digits=1)) s)")

    jldsave(joinpath(task.outdir, "chains", "samples.jld2");
            samples=samples, diag=diag)

    res = run_postprocess(
        samples, diag, task.label, task.outdir,
        task.n_burnin, task.y, task.P, task.df_clean,
        task.pc1, task.pc2, task.pct1, task.pct2;
        has_s   = task.has_s,
        c_fixed = task.c_fixed,
        color   = task.color,
        priors  = task.priors
    )

    return (
        scenario_id = task.scenario_id,
        tag         = task.tag,
        param       = Float64(something(task.param isa Real ? task.param : nothing, NaN)),
        label       = task.label,
        color       = task.color,
        k           = res.k,
        r           = res.r,
        α           = res.α,
        s           = res.s,
        ess_k       = res.ess_k,
        ess_r       = res.ess_r,
        ess_α       = res.ess_α,
        ess_s       = res.ess_s,
        has_s       = res.has_s,
        c_fixed     = res.c_fixed,
    )
end

# ============================================================================
# Data loading (master process only)
# ============================================================================

using XLSX

println("Loading GSI data...")
data_path = joinpath("data", "2023-Global-Slavery-Index-Data.xlsx")
df = DataFrame(XLSX.readtable(data_path, "GSI 2023 summary data"; first_row = 3))
rename!(df, Dict(
    "Estimated number of people in modern slavery" => :est_num_ms,
    "Population"                                   => :pop,
    "Governance issues"                            => :governance_issues,
    "Lack of basic needs"                          => :lack_basic_needs,
    "Inequality"                                   => :inequality,
    "Disenfranchised groups"                       => :disenfranchised_groups,
    "Effects of conflict"                          => :effects_of_conflict
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
D_eucl = pairwise(Euclidean(), X_std', dims=2)

n = length(y)
println("  n = $n observations")

# PCA for cluster visualisation (shared across scenarios)
F_svd    = svd(X_std)
pc1      = F_svd.U[:, 1] .* F_svd.S[1]
pc2      = F_svd.U[:, 2] .* F_svd.S[2]
var_expl = 100 .* F_svd.S .^ 2 ./ sum(F_svd.S .^ 2)
pct1     = round(var_expl[1], digits=1)
pct2     = round(var_expl[2], digits=1)

# ── Iteration counts ──────────────────────────────────────────────────────────
const N_SAMPLES = TEST_RUN ? 2_000  : 50_000
const N_BURNIN  = TEST_RUN ? 500    : 10_000
N_BURNIN < N_SAMPLES || error("N_BURNIN must be < N_SAMPLES")

# ── Precompute all distance matrices ─────────────────────────────────────────
const window_thresholds = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
const knn_values        = collect(1:10)

D_windows   = [window_distance_matrix(D_eucl, t) for t in window_thresholds]
D_knns      = [knn_distance_matrix(D_eucl, K)    for K in knn_values]
D_cosine    = pairwise(CosineDist(), X_std', dims=2)
D_manhattan = pairwise(Cityblock(), X_std', dims=2)

println("  Cosine dist:    mean=$(round(mean([D_cosine[i,j]    for i in 1:n, j in 1:n if i!=j]), digits=3))")
println("  Manhattan dist: mean=$(round(mean([D_manhattan[i,j] for i in 1:n, j in 1:n if i!=j]), digits=3))")

# ── DDCRPParams ───────────────────────────────────────────────────────────────
ddcrp_fixed_c = DDCRPParams(1.0, 1.0)
ddcrp_win_knn = DDCRPParams{Float64}(1.0, 1.0, identity_decay, 1.0, 0.01, nothing, nothing)
ddcrp_exp     = DDCRPParams(1.0, 1.0, 1.0, 0.01, 1.0, 0.01)

# ── MCMCOptions ───────────────────────────────────────────────────────────────
opts_fixed_c = MCMCOptions(
    n_samples         = N_SAMPLES,
    verbose           = VERBOSE,
    track_diagnostics = true,
    infer_params      = Dict(:r => true, :c => false),
    prop_sds          = Dict(:r => 0.5)
)

opts_win_knn = MCMCOptions(
    n_samples         = N_SAMPLES,
    verbose           = VERBOSE,
    track_diagnostics = true,
    infer_params      = Dict(:r => true, :c => true, :α_ddcrp => true),
    prop_sds          = Dict(:r => 0.5)
)

opts_exp = MCMCOptions(
    n_samples         = N_SAMPLES,
    verbose           = VERBOSE,
    track_diagnostics = true,
    infer_params      = Dict(:r => true, :c => true, :α_ddcrp => true, :s_ddcrp => true),
    prop_sds          = Dict(:r => 0.5, :s_ddcrp => 0.3)
)

# ── Scenario 1 init vectors ───────────────────────────────────────────────────
function build_c_twenty(n::Int)
    group_sz = max(1, n ÷ 20)
    c = Vector{Int}(undef, n)
    for i in 1:n
        g    = (i - 1) ÷ group_sz
        g    = min(g, 19)
        root = g * group_sz + 1
        c[i] = root
    end
    for g in 0:19
        root = g * group_sz + 1
        root <= n && (c[root] = root)
    end
    return c
end

c_one        = ones(Int, n)
c_twenty     = build_c_twenty(n)
c_singletons = collect(1:n)

# ── Shared task fields ────────────────────────────────────────────────────────
shared = (
    y        = y,
    P        = P,
    df_clean = df_clean,
    pc1      = pc1,
    pc2      = pc2,
    pct1     = pct1,
    pct2     = pct2,
    n_burnin = N_BURNIN,
    priors   = PRIORS,
)

# ============================================================================
# Build flat task list  (3 + 11 + 10 + 1 + 1 = 26 tasks)
# ============================================================================

mkpath("results/walkfree_testing")
tasks = []

# ── Scenario 1 ────────────────────────────────────────────────────────────────
mkpath("results/walkfree_testing/scenario1_fixed_c")
for (tag, c_init, col) in [
    ("one_cluster",     c_one,        :steelblue),
    ("twenty_clusters", c_twenty,     :darkorange),
    ("all_singletons",  c_singletons, :seagreen),
]
    push!(tasks, merge(shared, (
        scenario_id  = 1,
        tag          = tag,
        param        = nothing,
        label        = "Sc1/$tag",
        outdir       = "results/walkfree_testing/scenario1_fixed_c/$tag",
        D            = D_eucl,
        ddcrp_params = ddcrp_fixed_c,
        opts         = opts_fixed_c,
        init_c       = c_init,
        has_s        = false,
        c_fixed      = true,
        color        = col,
    )))
end

# ── Scenario 2 — Window distance sweep ───────────────────────────────────────
mkpath("results/walkfree_testing/scenario2_window")
for (i, thresh) in enumerate(window_thresholds)
    tag = replace("$(thresh)", "." => "p")
    push!(tasks, merge(shared, (
        scenario_id  = 2,
        tag          = "window_$tag",
        param        = thresh,
        label        = "Window=$thresh",
        outdir       = "results/walkfree_testing/scenario2_window/window_$tag",
        D            = D_windows[i],
        ddcrp_params = ddcrp_win_knn,
        opts         = opts_win_knn,
        init_c       = nothing,
        has_s        = false,
        c_fixed      = false,
        color        = :steelblue,
    )))
end

# ── Scenario 3 — KNN sweep ────────────────────────────────────────────────────
mkpath("results/walkfree_testing/scenario3_knn")
for (i, K) in enumerate(knn_values)
    push!(tasks, merge(shared, (
        scenario_id  = 3,
        tag          = "knn_$K",
        param        = K,
        label        = "KNN K=$K",
        outdir       = "results/walkfree_testing/scenario3_knn/knn_$K",
        D            = D_knns[i],
        ddcrp_params = ddcrp_win_knn,
        opts         = opts_win_knn,
        init_c       = nothing,
        has_s        = false,
        c_fixed      = false,
        color        = :darkorange,
    )))
end

# ── Scenario 4 — Cosine + Exponential ────────────────────────────────────────
mkpath("results/walkfree_testing/scenario4_cosine")
push!(tasks, merge(shared, (
    scenario_id  = 4,
    tag          = "cosine",
    param        = nothing,
    label        = "Cosine+Exp",
    outdir       = "results/walkfree_testing/scenario4_cosine",
    D            = D_cosine,
    ddcrp_params = ddcrp_exp,
    opts         = opts_exp,
    init_c       = nothing,
    has_s        = true,
    c_fixed      = false,
    color        = :purple,
)))

# ── Scenario 5 — Manhattan + Exponential ─────────────────────────────────────
mkpath("results/walkfree_testing/scenario5_manhattan")
push!(tasks, merge(shared, (
    scenario_id  = 5,
    tag          = "manhattan",
    param        = nothing,
    label        = "Manhattan+Exp",
    outdir       = "results/walkfree_testing/scenario5_manhattan",
    D            = D_manhattan,
    ddcrp_params = ddcrp_exp,
    opts         = opts_exp,
    init_c       = nothing,
    has_s        = true,
    c_fixed      = false,
    color        = :crimson,
)))

println("\nBuilt $(length(tasks)) tasks across 5 scenarios.")
println("Running with $(nworkers()) worker process(es)...\n")

# ============================================================================
# Run all tasks in parallel via pmap
# ============================================================================

results_all = pmap(run_scenario_task, tasks)

# ============================================================================
# Cross-scenario comparison plots (master only, sequential)
# ============================================================================

println("\n" * "="^70)
println("Cross-scenario comparisons")
println("="^70)

# ── Scenario 1: r density overlay ────────────────────────────────────────────
results_s1 = filter(r -> r.scenario_id == 1, results_all)
p_r_cmp = plot(title="Posterior r — Scenario 1 sub-runs",
               xlabel="r", ylabel="Density", size=(700, 400))
for res in results_s1
    density!(p_r_cmp, res.r; label=res.tag, color=res.color, linewidth=2)
end
savefig(p_r_cmp, "results/walkfree_testing/scenario1_fixed_c/r_density_comparison.png")
println("Scenario 1 comparison saved.")

# ── Scenario 2: K posterior overlay + mean K vs threshold ────────────────────
results_s2 = sort(filter(r -> r.scenario_id == 2, results_all), by=r -> r.param)
palette_s2 = cgrad(:viridis, length(results_s2); categorical=true)

p_k_win = plot(title="Posterior K — Window threshold sweep",
               xlabel="K", ylabel="Probability", size=(900, 500), legend=:outerright)
for (i, res) in enumerate(results_s2)
    cm_k = countmap(res.k)
    ks   = sort(collect(keys(cm_k)))
    ps   = [cm_k[k] / length(res.k) for k in ks]
    plot!(p_k_win, ks, ps; label="t=$(res.param)", color=palette_s2[i],
          markershape=:circle, linewidth=1.5)
end
savefig(p_k_win, "results/walkfree_testing/scenario2_window/k_distribution_comparison.png")

thresholds_vec = [r.param   for r in results_s2]
mean_k_win     = [mean(r.k) for r in results_s2]
p_mk_win = plot(thresholds_vec, mean_k_win;
                title="Mean K vs Window threshold",
                xlabel="Threshold", ylabel="Mean K",
                markershape=:circle, linewidth=2, legend=false, color=:steelblue)
savefig(p_mk_win, "results/walkfree_testing/scenario2_window/mean_K_vs_threshold.png")

summary_s2 = DataFrame(
    threshold  = thresholds_vec,
    mean_K     = [round(mean(r.k), digits=3)  for r in results_s2],
    median_K   = [median(r.k)                  for r in results_s2],
    mode_K     = [argmax(countmap(r.k))         for r in results_s2],
    mean_r     = [round(mean(r.r), digits=4)  for r in results_s2],
    mean_alpha = [round(mean(r.α), digits=4)  for r in results_s2],
    ess_K      = [round(r.ess_k, digits=1)    for r in results_s2],
)
CSV.write("results/walkfree_testing/scenario2_window/summary_comparison.csv", summary_s2)
println("Scenario 2 comparisons saved.")

# ── Scenario 3: K posterior overlay + mean K vs KNN K ────────────────────────
results_s3 = sort(filter(r -> r.scenario_id == 3, results_all), by=r -> r.param)
palette_s3 = cgrad(:plasma, length(results_s3); categorical=true)

p_k_knn = plot(title="Posterior K — KNN sweep",
               xlabel="K", ylabel="Probability", size=(900, 500), legend=:outerright)
for (i, res) in enumerate(results_s3)
    cm_k = countmap(res.k)
    ks   = sort(collect(keys(cm_k)))
    ps   = [cm_k[k] / length(res.k) for k in ks]
    plot!(p_k_knn, ks, ps; label="KNN=$(Int(res.param))", color=palette_s3[i],
          markershape=:circle, linewidth=1.5)
end
savefig(p_k_knn, "results/walkfree_testing/scenario3_knn/k_distribution_comparison.png")

knn_vec    = [Int(r.param) for r in results_s3]
mean_k_knn = [mean(r.k)    for r in results_s3]
p_mk_knn = plot(knn_vec, mean_k_knn;
                title="Mean K vs KNN K",
                xlabel="K (nearest neighbours)", ylabel="Mean K",
                markershape=:circle, linewidth=2, legend=false, color=:darkorange)
savefig(p_mk_knn, "results/walkfree_testing/scenario3_knn/mean_K_vs_knn.png")

summary_s3 = DataFrame(
    knn_K      = knn_vec,
    mean_K     = [round(mean(r.k), digits=3)  for r in results_s3],
    median_K   = [median(r.k)                  for r in results_s3],
    mode_K     = [argmax(countmap(r.k))         for r in results_s3],
    mean_r     = [round(mean(r.r), digits=4)  for r in results_s3],
    mean_alpha = [round(mean(r.α), digits=4)  for r in results_s3],
    ess_K      = [round(r.ess_k, digits=1)    for r in results_s3],
)
CSV.write("results/walkfree_testing/scenario3_knn/summary_comparison.csv", summary_s3)
println("Scenario 3 comparisons saved.")

println("\n" * "="^70)
println("All scenarios complete.  Results in results/walkfree_testing/")
println("="^70)
