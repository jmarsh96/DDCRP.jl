# ============================================================================
# Old Faithful Eruption Data — ddCRP Analysis
# ============================================================================
#
# Applies GammaClusterShapeMarg to Old Faithful geyser eruption data.
#
# Model:
#   y_i | α_k, β_k  ~  Gamma(α_k, β_k)   (eruption duration in minutes)
#   β_k             ~  Gamma(β_a, β_b)    (marginalised out analytically)
#   α_k             ~  Gamma(α_a, α_b)    (sampled via MH on log-scale)
#
# Distance: d_{ij} = |w_i - w_j|  (absolute waiting time difference)
# ddCRP:    exponential decay; concentration α and scale selected by CV
#
# Birth proposal:     InverseGammaMomentMatch (falls back to prior for shape)
# Fixed-dim proposal: Resample(InverseGammaMomentMatch)
# ============================================================================

using DDCRP
using Random, Distributions
using StatsPlots, Plots
using DataFrames, CSV, Distances
using Statistics, StatsBase
using Printf

Random.seed!(2025)

# ============================================================================
# 0. Output directories
# ============================================================================

mkpath("results/old_faithful/figures")

# ============================================================================
# 1. Load data
# ============================================================================

println("Loading Old Faithful data...")
faithful = DataFrame(CSV.File(joinpath("data", "faithful.csv")))
y = Float64.(faithful.eruptions)   # response: eruption duration (min)
w = Float64.(faithful.waiting)     # covariate: waiting time (min)
n = length(y)
println("  n = $n observations")
println("  Eruption duration: [$(round(minimum(y), digits=2)), $(round(maximum(y), digits=2))] min")
println("  Waiting time:      [$(minimum(w)), $(maximum(w))] min")

# ============================================================================
# 2. Distance matrix: d_{ij} = |w_i - w_j|
# ============================================================================

D = pairwise(Euclidean(), w)   # n×n matrix of absolute waiting-time differences

# ============================================================================
# 3. Model and priors
# ============================================================================

model  = GammaClusterShapeMarg()
priors = GammaClusterShapeMargPriors(α_a=2.0, α_b=0.5, β_a=2.0, β_b=0.5)

# ============================================================================
# 4. Proposals
# ============================================================================

birth_proposal     = InverseGammaMomentMatch()
fixed_dim_proposal = Resample(InverseGammaMomentMatch())

# ============================================================================
# 5. Helper functions for CV evaluation
# ============================================================================

# Numerically stable log-sum-exp
stable_logsumexp(xs) = (m = maximum(xs); m + log(sum(exp.(xs .- m))))

# Compute n_post × n matrix of log p(y_i | θ^(t)) for the Gamma model.
# β is marginalised in the sampler; here we recover it by sampling from the
# posterior Gamma(β_a + n_k*α_k, 1/(β_b + Σy_k)) at each iteration.
function gamma_loglik_matrix(
    samp,
    y_obs::Vector{Float64},
    pr::GammaClusterShapeMargPriors,
    post_idx::AbstractRange
)
    n_obs  = length(y_obs)
    n_post = length(post_idx)
    ll = Matrix{Float64}(undef, n_post, n_obs)
    for (t, iter) in enumerate(post_idx)
        c_cur  = samp.c[iter, :]
        tables = table_vector(c_cur)
        for table in tables
            S_k = sum(y_obs[table])
            n_k = length(table)
            for i in table
                α_i = samp.α[iter, i]
                β_i = rand(Gamma(pr.β_a + n_k * α_i, 1 / (pr.β_b + S_k)))
                ll[t, i] = logpdf(Gamma(α_i, 1 / β_i), y_obs[i])
            end
        end
    end
    return ll
end

# WAIC from log-likelihood matrix (lower is better).
function gamma_waic(ll::Matrix{Float64})
    n_post, n_obs = size(ll)
    lppd   = sum(stable_logsumexp(ll[:, i]) - log(n_post) for i in 1:n_obs)
    p_waic = sum(var(ll[:, i]) for i in 1:n_obs)
    return (waic = -2 * (lppd - p_waic), lppd = lppd, p_waic = p_waic)
end

# LPML via harmonic-mean CPO estimator (higher is better).
function gamma_lpml(ll::Matrix{Float64})
    n_post, n_obs = size(ll)
    return sum(-(stable_logsumexp(-ll[:, i]) - log(n_post)) for i in 1:n_obs)
end

# Held-out log-predictive density for test observations given training posterior.
#
# For each test obs j and posterior sample s the predictive is a mixture over
# training clusters, weighted by ddCRP link probabilities, plus a self-link
# term (new cluster) evaluated at the prior predictive.
#
# ddCRP link weight from test obs j to training obs i:
#   f_i = exp(-|w_j - w_train_i| * scale)
# Cluster-k weight: Σ_{i∈k} f_i / (Σ_all f + ddcrp_α)
# Self-link weight: ddcrp_α / (Σ_all f + ddcrp_α)
function held_out_logpred(
    test_idx::Vector{Int},
    train_idx::Vector{Int},
    train_samp,
    y_obs::Vector{Float64},
    w_obs::Vector{Float64},
    pr::GammaClusterShapeMargPriors,
    post_idx::AbstractRange,
    ddcrp_α::Float64,
    scale::Float64
)
    w_train  = w_obs[train_idx]
    y_train  = y_obs[train_idx]
    n_post   = length(post_idx)
    total_lp = 0.0

    for j in test_idx
        w_j = w_obs[j]
        y_j = y_obs[j]

        # ddCRP link weights from test obs j to each training obs
        f          = [exp(-abs(w_j - w_train[i]) * scale) for i in eachindex(w_train)]
        total_mass = sum(f) + ddcrp_α

        # Per-sample log-predictive for obs j
        lp_s = Vector{Float64}(undef, n_post)
        for (t, iter) in enumerate(post_idx)
            c_cur  = train_samp.c[iter, :]
            tables = table_vector(c_cur)
            log_terms = Float64[]

            for table in tables
                # Cluster weight = sum of ddCRP link weights to its members
                w_k = sum(f[table]) / total_mass
                w_k <= 0.0 && continue

                n_k = length(table)
                S_k = sum(y_train[table])
                α_k = train_samp.α[iter, first(table)]
                β_k = rand(Gamma(pr.β_a + n_k * α_k, 1 / (pr.β_b + S_k)))
                push!(log_terms, log(w_k) + logpdf(Gamma(α_k, 1 / β_k), y_j))
            end

            # Self-link: new cluster drawn from prior
            w_self = ddcrp_α / total_mass
            α_new  = rand(Gamma(pr.α_a, 1 / pr.α_b))
            β_new  = rand(Gamma(pr.β_a, 1 / pr.β_b))
            push!(log_terms, log(w_self) + logpdf(Gamma(α_new, 1 / β_new), y_j))

            lp_s[t] = stable_logsumexp(log_terms)
        end

        total_lp += stable_logsumexp(lp_s) - log(n_post)
    end
    return total_lp
end

# ============================================================================
# 6. Cross-validation: select ddCRP concentration α and decay scale
# ============================================================================

println("\nRunning cross-validation over ddCRP hyperparameters...")

α_grid     = [0.5, 1.0, 2.0, 5.0, 10.0]
scale_grid = [0.01, 0.02, 0.05, 0.1, 0.2]

n_cv        = 5_000
cv_burnin   = 1_000
cv_post_idx = (cv_burnin + 1):n_cv

cv_opts = MCMCOptions(
    n_samples         = n_cv,
    verbose           = false,
    infer_params      = Dict(:α => true, :c => true),
    prop_sds          = Dict(:α => 0.5),
    track_diagnostics = false
)

# Fixed 80/20 train/test split (separate RNG keeps main seed unperturbed)
Random.seed!(42)
train_idx = sort(sample(1:n, round(Int, 0.8 * n), replace=false))
test_idx  = setdiff(1:n, train_idx)
D_train   = D[train_idx, train_idx]
println("  Train n=$(length(train_idx)), Test n=$(length(test_idx))")

cv_results = NamedTuple[]
println()
@printf "  %-6s  %-6s  %9s  %9s  %9s  %7s\n" "α" "scale" "WAIC" "LPML" "HOLL" "mean_K"
println("  " * "-"^57)

for α_cv in α_grid, sc_cv in scale_grid
    params_cv = DDCRPParams(Float64(α_cv), Float64(sc_cv))

    # --- In-sample WAIC / LPML (full data) ---
    s_full = mcmc(model, ContinuousData(y, D), params_cv, priors,
                  birth_proposal;
                  fixed_dim_proposal = fixed_dim_proposal,
                  opts               = cv_opts)
    ll       = gamma_loglik_matrix(s_full, y, priors, cv_post_idx)
    waic_v   = gamma_waic(ll).waic
    lpml_v   = gamma_lpml(ll)
    mean_k_v = mean(calculate_n_clusters(s_full.c[cv_post_idx, :]))

    # --- Held-out log-predictive (train/test split) ---
    s_train = mcmc(model, ContinuousData(y[train_idx], D_train), params_cv, priors,
                   birth_proposal;
                   fixed_dim_proposal = fixed_dim_proposal,
                   opts               = cv_opts)
    holl_v = held_out_logpred(test_idx, train_idx, s_train, y, w, priors,
                               cv_post_idx, Float64(α_cv), Float64(sc_cv))

    push!(cv_results, (α      = Float64(α_cv),
                       scale  = Float64(sc_cv),
                       waic   = waic_v,
                       lpml   = lpml_v,
                       holl   = holl_v,
                       mean_k = mean_k_v))
    @printf "  %-6.2f  %-6.3f  %9.2f  %9.2f  %9.2f  %7.2f\n" α_cv sc_cv waic_v lpml_v holl_v mean_k_v
end

# Select optimal parameters by held-out log-likelihood (higher is better)
best_cv   = argmax(r -> r.holl, cv_results)
α_opt     = best_cv.α
scale_opt = best_cv.scale
println()
println("  Optimal: α=$α_opt, scale=$scale_opt")
println("    WAIC=$(round(best_cv.waic, digits=2))  LPML=$(round(best_cv.lpml, digits=2))  HOLL=$(round(best_cv.holl, digits=2))")

# Save CV grid CSV
cv_df = DataFrame(cv_results)
CSV.write("results/old_faithful/cv_grid.csv", cv_df)
println("  CV grid saved to results/old_faithful/cv_grid.csv")

# CV heatmap figures (scale on x-axis, α on y-axis)
n_α     = length(α_grid)
n_scale = length(scale_grid)
# Loop order: outer α_grid, inner scale_grid → reshape fills scale first
waic_mat = reshape([r.waic for r in cv_results], n_scale, n_α)
lpml_mat = reshape([r.lpml for r in cv_results], n_scale, n_α)
holl_mat = reshape([r.holl for r in cv_results], n_scale, n_α)

p_cv_waic = heatmap(scale_grid, α_grid, waic_mat',
    xlabel="scale", ylabel="α", title="WAIC surface (lower = better)",
    color=cgrad(:viridis, rev=true), colorbar_title="WAIC")
savefig(p_cv_waic, "results/old_faithful/figures/cv_waic.png")

p_cv_lpml = heatmap(scale_grid, α_grid, lpml_mat',
    xlabel="scale", ylabel="α", title="LPML surface (higher = better)",
    color=:viridis, colorbar_title="LPML")
savefig(p_cv_lpml, "results/old_faithful/figures/cv_lpml.png")

p_cv_holl = heatmap(scale_grid, α_grid, holl_mat',
    xlabel="scale", ylabel="α", title="Held-out log-likelihood (higher = better)",
    color=:viridis, colorbar_title="HOLL")
savefig(p_cv_holl, "results/old_faithful/figures/cv_holl.png")

ddcrp_params = DDCRPParams(α_opt, scale_opt)

# ============================================================================
# 7. MCMC options
# ============================================================================

n_samples = 20_000
n_burnin  = 5_000

opts = MCMCOptions(
    n_samples         = n_samples,
    verbose           = true,
    infer_params      = Dict(:α => true, :c => true),
    prop_sds          = Dict(:α => 0.5),
    track_diagnostics = true
)

# ============================================================================
# 8. Run MCMC
# ============================================================================

println("\nRunning RJMCMC (n=$n, $n_samples iterations, $n_burnin burn-in)...")
println("  ddCRP: α=$α_opt, scale=$scale_opt  (selected by CV)")
Random.seed!(123)
samples, diagnostics = mcmc(
    model, ContinuousData(y, D), ddcrp_params, priors,
    birth_proposal;
    fixed_dim_proposal = fixed_dim_proposal,
    opts               = opts
)
println("  Done. Total time: $(round(diagnostics.total_time, digits=1)) s")

# ============================================================================
# 9. Post-processing
# ============================================================================

post_idx = (n_burnin + 1):n_samples
c_post   = samples.c[post_idx, :]
lp_post  = samples.logpost[post_idx]
k_post   = calculate_n_clusters(c_post)
sim      = compute_similarity_matrix(c_post)

ar     = acceptance_rates(diagnostics)
ess_k  = effective_sample_size(Float64.(k_post))
ess_lp = effective_sample_size(lp_post)

# ESS for shape parameters: one trace per observation, report median and min
α_post    = samples.α[post_idx, :]
ess_α_vec = [effective_sample_size(α_post[:, i]) for i in 1:n]
ess_α_med = median(ess_α_vec)
ess_α_min = minimum(ess_α_vec)

println()
println("="^54)
println("  Summary — post burn-in ($(length(post_idx)) samples)")
println("="^54)
@printf "  %-32s  %s\n"   "Metric"                      "Value"
println("-"^54)
@printf "  %-32s  %6.3f\n" "Birth acceptance rate"       ar.birth
@printf "  %-32s  %6.3f\n" "Death acceptance rate"       ar.death
@printf "  %-32s  %6.3f\n" "Fixed-dim acceptance rate"   ar.fixed
println("-"^54)
@printf "  %-32s  %6.1f\n" "ESS  K"                      ess_k
@printf "  %-32s  %6.1f\n" "ESS  log-posterior"          ess_lp
@printf "  %-32s  %6.1f\n" "ESS  α  (median over obs)"   ess_α_med
@printf "  %-32s  %6.1f\n" "ESS  α  (min over obs)"      ess_α_min
println("-"^54)
@printf "  %-32s  %6.2f\n" "K  mean"                     mean(k_post)
@printf "  %-32s  %6d\n"   "K  median"                   Int(median(k_post))
@printf "  %-32s  %6d\n"   "K  mode"                     argmax(countmap(k_post))
println("="^54)

# MAP clustering: iteration with highest log-posterior
map_iter    = argmax(lp_post)
c_map       = c_post[map_iter, :]
cluster_ids = compute_table_assignments(c_map)
println("MAP clustering: $(length(unique(cluster_ids))) clusters")

# ============================================================================
# 10. Figures
# ============================================================================

# 10a. Trace plots
p_k = plot(k_post,
    xlabel="Post burn-in iteration", ylabel="K",
    title="Number of clusters", legend=false, linewidth=0.5)
p_lp = plot(lp_post,
    xlabel="Post burn-in iteration", ylabel="Log-posterior",
    title="Log-posterior trace", legend=false, linewidth=0.5)
p_traces = plot(p_k, p_lp, layout=(1, 2), size=(1200, 400))
savefig(p_traces, "results/old_faithful/figures/traces.png")

# 10b. K distribution
cm_k    = countmap(k_post)
k_vals  = sort(collect(keys(cm_k)))
k_probs = [cm_k[k] / length(k_post) for k in k_vals]
p_kdist = bar(k_vals, k_probs,
    xlabel="K", ylabel="Posterior probability",
    title="Posterior distribution of K",
    legend=false, color=:steelblue)
savefig(p_kdist, "results/old_faithful/figures/k_distribution.png")

# 10c. Scatter: waiting time vs eruption duration, coloured by MAP clustering
p_scatter = scatter(w, y,
    group=cluster_ids,
    xlabel="Waiting time (min)", ylabel="Eruption duration (min)",
    title="Old Faithful: MAP cluster assignments",
    markersize=4, palette=:tab10, legend=:topleft)
savefig(p_scatter, "results/old_faithful/figures/clustering_scatter.png")

# 10d. Co-clustering network: arrows between high-probability pairs
# Sort observations by waiting time so within-cluster arrows are short and non-crossing.
k_top = 5   # links per observation to consider

ord    = sortperm(w)
w_s    = w[ord]
y_s    = y[ord]
sim_s  = sim[ord, ord]
cids_s = cluster_ids[ord]

p_network = scatter(w_s, y_s, group=cids_s,
    xlabel="Waiting time (min)", ylabel="Eruption duration (min)",
    title="Co-clustering network (top-$k_top links per observation)",
    markersize=4, palette=:tab10, legend=false, size=(900, 600))

drawn = Set{Tuple{Int,Int}}()
for i in 1:n
    row    = copy(sim_s[i, :])
    row[i] = -Inf                            # exclude self
    top_js = sortperm(row, rev=true)[1:k_top]
    for j in top_js
        p_ij = sim_s[i, j]
        p_ij <= 0.1 && continue              # skip negligible links
        edge = (min(i, j), max(i, j))
        edge ∈ drawn && continue             # draw each pair once
        push!(drawn, edge)
        plot!(p_network,
            [w_s[i], w_s[j]], [y_s[i], y_s[j]],
            arrow     = arrow(:both, :closed, 0.15, 0.15),
            color     = :steelblue,
            alpha     = clamp(p_ij * 0.7, 0.05, 0.7),
            linewidth = clamp(p_ij * 1.5, 0.3, 1.5),
            label     = false)
    end
end

# Redraw points on top so they sit above the arrow lines
scatter!(p_network, w_s, y_s, group=cids_s,
    markersize=5, palette=:tab10, legend=false)
savefig(p_network, "results/old_faithful/figures/coclustering_network.png")

# 10e. Posterior predictive check
println("\nComputing posterior predictive distribution...")
n_post  = length(post_idx)
ppd     = Matrix{Float64}(undef, n, n_post)

for (t, iter) in enumerate(post_idx)
    c_cur  = samples.c[iter, :]
    tables = table_vector(c_cur)
    for table in tables
        S_k = sum(y[table])
        n_k = length(table)
        for i in table
            α_i   = samples.α[iter, i]
            β_i   = rand(Gamma(priors.β_a + n_k * α_i, 1 / (priors.β_b + S_k)))
            ppd[i, t] = rand(Gamma(α_i, 1 / β_i))
        end
    end
end

ppd_means = vec(mean(ppd, dims=2))
ppd_lower = vec(mapslices(col -> quantile(col, 0.025), ppd, dims=2))
ppd_upper = vec(mapslices(col -> quantile(col, 0.975), ppd, dims=2))

p_ppd = scatter(1:n, y, label="Observed", color=:red, markersize=3,
    xlabel="Observation index", ylabel="Eruption duration (min)",
    title="Posterior Predictive Check", legend=:topright)
scatter!(p_ppd, 1:n, ppd_means, label="PPD mean", color=:blue, markersize=2)
for i in 1:n
    plot!(p_ppd, [i, i], [ppd_lower[i], ppd_upper[i]],
        color=:blue, alpha=0.2, label=(i == 1 ? "95% CI" : false))
end
plot!(p_ppd, size=(1000, 500))
savefig(p_ppd, "results/old_faithful/figures/ppd_check.png")

println("Figures saved to results/old_faithful/figures/")

# ============================================================================
# 11. Summary metrics CSV
# ============================================================================

metrics = DataFrame(
    metric = [
        "Birth acceptance rate",
        "Death acceptance rate",
        "Fixed-dim acceptance rate",
        "ESS (K)",
        "ESS (log-posterior)",
        "ESS (α, median over obs)",
        "ESS (α, min over obs)",
        "K mean",
        "K median",
        "K mode",
        "Total time (s)",
        "CV optimal α",
        "CV optimal scale",
    ],
    value = [
        round(ar.birth,               digits=3),
        round(ar.death,               digits=3),
        round(ar.fixed,               digits=3),
        round(ess_k,                  digits=1),
        round(ess_lp,                 digits=1),
        round(ess_α_med,              digits=1),
        round(ess_α_min,              digits=1),
        round(mean(k_post),           digits=2),
        Float64(Int(median(k_post))),
        Float64(argmax(countmap(k_post))),
        round(diagnostics.total_time, digits=1),
        α_opt,
        scale_opt,
    ]
)
CSV.write("results/old_faithful/summary_metrics.csv", metrics)
println("Summary metrics saved to results/old_faithful/summary_metrics.csv")

# ============================================================================
# 12. LaTeX table output
# ============================================================================
#
# Writes two .tex files to results/old_faithful/tables/ that can be \input{}'d
# directly into docs/paper_draft.tex. Re-run this script to regenerate after
# new results are available.

mkpath("results/old_faithful/tables")


function cv_top_to_latex(df::DataFrame, α_opt::Float64, scale_opt::Float64,
                          filepath::String; top_n::Int=10)
    sorted = sort(df, :holl, rev=true)
    top    = first(sorted, top_n)

    open(filepath, "w") do io
        println(io, raw"\begin{table}[htbp]")
        println(io, raw"\centering")
        println(io, raw"\caption{Top cross-validation results for the Old Faithful analysis,")
        println(io, raw"  ranked by held-out log-likelihood (HOLL, higher is better).")
        println(io, raw"  WAIC (lower is better) and LPML (higher is better) use the full dataset;")
        println(io, raw"  HOLL uses an 80/20 train/test split.}")
        println(io, raw"  The bold row indicates the selected hyperparameters.")
        println(io, raw"\label{tab:of_cv}")
        println(io, raw"\begin{tabular}{rrrrrr}")
        println(io, raw"\toprule")
        println(io, raw"$\alpha$ & $s$ & WAIC & LPML & HOLL & $\bar{K}$ \\\\")
        println(io, raw"\hline")

        for row in eachrow(top)
            is_best = (row.α == α_opt && row.scale == scale_opt)
            fields = [
                string(row.α),
                string(row.scale),
                string(round(row.waic,   digits=2)),
                string(round(row.lpml,   digits=2)),
                string(round(row.holl,   digits=2)),
                string(round(row.mean_k, digits=2)),
            ]
            line = join(fields, " & ") * " \\\\"
            println(io, is_best ? "\\textbf{$(join(fields, "} & \\textbf{"))} \\\\" : line)
        end

        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
        println(io, raw"\end{table}")
    end
end

cv_top_to_latex(cv_df, α_opt, scale_opt, "results/old_faithful/tables/cv_top.tex")
println("LaTeX tables saved to results/old_faithful/tables/")
println("\nDone.")
