using DDCRP
using CSV, DataFrames, Distances
using Statistics, StatsBase, LinearAlgebra
using Random
using Printf
using XLSX
using JLD2
using Distributions
using SpecialFunctions
using StatsPlots

data_path = joinpath("data", "GSI_with_survey.xlsx")
df = DataFrame(XLSX.readtable(data_path, "GSI 2023 summary data"))
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

covariate_cols = [
    :governance_issues,
    :lack_basic_needs,
    :inequality,
    :disenfranchised_groups,
    :effects_of_conflict
]
required_cols = [:est_num_ms, :pop, covariate_cols...]
df_clean = dropmissing(df, required_cols)

y = ifelse.(df_clean.Surveyed, df_clean.est_num_ms, missing)
P = Int.(df_clean.pop)

X = Float64.(Matrix(df_clean[:, covariate_cols]))
X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)
D = pairwise(Euclidean(), X_std', dims=2)
n = length(y)

# Pick a missing observation index for comparison
missing_idx = findfirst(ismissing, y)
@printf("Comparing models on %d observations, %d missing. Tracking index %d.\n",
        n, sum(ismissing.(y)), missing_idx)


ddcrp_params = DDCRPParams(1.0, 1.0, 1.0, 0.1, 1.0, 0.1)
n_samples = 250_000
#n_samples = 50_000
opts_common = MCMCOptions(
    n_samples = n_samples,
    verbose = true,
    infer_params = Dict(
        :s_ddcrp => false,
        :r => true,
        :α_ddcrp => true
    ),
    track_diagnostics = true
)

# ============================================================
# Model 1: NBPopulationRatesMarg (Gibbs via ConjugateProposal)
# ============================================================
println("\n=== NBPopulationRatesMarg (Gibbs) ===")
model_marg  = NBPopulationRatesMarg()
γ_a = 1.0
γ_b = 0.1
r_a = 1.0
r_b = 0.1
priors_marg = NBPopulationRatesMargPriors(γ_a, γ_b, r_a, r_b)

samples_marg, _ = mcmc(
    model_marg, y, P, D, ddcrp_params, priors_marg, ConjugateProposal();
    opts = opts_common,
    init_params = Dict{Symbol,Any}(:r => 10.0)
)

# ============================================================
# Model 2: NBPopulationRates (RJMCMC via PriorProposal)
# ============================================================
println("\n=== NBPopulationRates (RJMCMC) ===")
model_rjmcmc  = NBPopulationRates()
priors_rjmcmc = NBPopulationRatesPriors(γ_a, γ_b, r_a, r_b)
proposal = FixedDistributionProposal([Exponential(10.0)])
samples_rjmcmc, _ = mcmc(
    model_rjmcmc, y, P, D, ddcrp_params, priors_rjmcmc, proposal;
    opts = opts_common,
    init_params = Dict{Symbol,Any}(:r => 10.0)
)

# ============================================================
# Comparison plots
# ============================================================
burnin = div(n_samples, 4)
idx    = (burnin+1):n_samples

p1 = plot(
    calculate_n_clusters(samples_marg.c[idx, :]),
    label="Gibbs (Marg)", alpha=0.7, title="Number of clusters", xlabel="Iteration"
)
plot!(p1,
    calculate_n_clusters(samples_rjmcmc.c[idx, :]),
    label="RJMCMC", alpha=0.7
)

p2 = density(samples_marg.r[idx],   label="Gibbs (Marg)", title="r (dispersion)")
density!(p2, samples_rjmcmc.r[idx], label="RJMCMC")

p3 = density(samples_marg.α_ddcrp,   label="Gibbs (Marg)",
             title="α_ddcrp")
density!(p3, samples_rjmcmc.α_ddcrp, label="RJMCMC")

p4 = plot(samples_marg.logpost[idx],   label="Gibbs (Marg)", title="Log-posterior", alpha=0.7)
plot!(p4, samples_rjmcmc.logpost[idx], label="RJMCMC", alpha=0.7)

display(plot(p1, p2, p3, p4, layout=(2,2), size=(900,600)))

# ============================================================
# Grouped barplot: posterior distribution of n_clusters
# ============================================================
nc_marg   = calculate_n_clusters(samples_marg.c[idx, :])
nc_rjmcmc = calculate_n_clusters(samples_rjmcmc.c[idx, :])
cm_marg   = countmap(nc_marg)
cm_rjmcmc = countmap(nc_rjmcmc)
all_k     = sort(collect(union(keys(cm_marg), keys(cm_rjmcmc))))
n_idx     = length(idx)

pct_marg   = [100 * get(cm_marg,   k, 0) / n_idx for k in all_k]
pct_rjmcmc = [100 * get(cm_rjmcmc, k, 0) / n_idx for k in all_k]

p_bar = groupedbar(
    all_k,
    hcat(pct_marg, pct_rjmcmc),
    bar_position = :dodge,
    bar_width     = 0.7,
    label         = ["Gibbs (Marg)" "RJMCMC"],
    xlabel        = "Number of clusters",
    ylabel        = "Posterior probability (%)",
    title         = "Posterior distribution of n_clusters",
    legend        = :topright,
    size          = (1600, 800)
)
display(p_bar)

# ============================================================
# Numerical summaries
# ============================================================
println("\n--- Posterior summaries (post-burnin) ---")
@printf("%-30s %10s %10s\n", "Quantity", "Gibbs", "RJMCMC")

nc_marg   = calculate_n_clusters(samples_marg.c[idx, :])
nc_rjmcmc = calculate_n_clusters(samples_rjmcmc.c[idx, :])

@printf("%-30s %10.2f %10.2f\n", "mean(n_clusters)",
        mean(nc_marg), mean(nc_rjmcmc))
@printf("%-30s %10.2f %10.2f\n", "std(n_clusters)",
        std(nc_marg), std(nc_rjmcmc))
@printf("%-30s %10d %10d\n", "median(n_clusters)",
        Int(median(nc_marg)), Int(median(nc_rjmcmc)))
@printf("%-30s %10d %10d\n", "mode(n_clusters)",
        mode(nc_marg), mode(nc_rjmcmc))
@printf("%-30s  [%d, %d]   [%d, %d]\n", "95%% CI (n_clusters)",
        Int(quantile(nc_marg, 0.025)), Int(quantile(nc_marg, 0.975)),
        Int(quantile(nc_rjmcmc, 0.025)), Int(quantile(nc_rjmcmc, 0.975)))
println("\n--- Posterior distribution of n_clusters ---")
cm_marg   = countmap(nc_marg)
cm_rjmcmc = countmap(nc_rjmcmc)
all_k = sort(collect(union(keys(cm_marg), keys(cm_rjmcmc))))
n_idx = length(idx)
@printf("%-10s %12s %12s\n", "n_clusters", "Gibbs (%)", "RJMCMC (%)")
for k in all_k
    p_marg   = 100 * get(cm_marg,   k, 0) / n_idx
    p_rjmcmc = 100 * get(cm_rjmcmc, k, 0) / n_idx
    @printf("%-10d %12.2f %12.2f\n", k, p_marg, p_rjmcmc)
end

@printf("%-30s %10.4f %10.4f\n", "mean(r)",
        mean(samples_marg.r[idx]), mean(samples_rjmcmc.r[idx]))
@printf("%-30s %10.4f %10.4f\n", "std(r)",
        std(samples_marg.r[idx]),  std(samples_rjmcmc.r[idx]))

# ============================================================
# PCA scatter: PC1 vs PC2, coloured by MAP clusters
# ============================================================
# PCA via SVD on standardised covariates
U, S, V = svd(X_std)
pc1 = X_std * V[:, 1]
pc2 = X_std * V[:, 2]

z_marg   = point_estimate_clustering(samples_marg.c[idx, :];   method=:median_K)
z_rjmcmc = point_estimate_clustering(samples_rjmcmc.c[idx, :]; method=:median_K)

println("\n--- Cluster assignments (median_K point estimate) ---")
println("\nGibbs (Marg):")
for k in sort(unique(z_marg))
    members = df_clean.Country[z_marg .== k]
    @printf("  Cluster %d (%d members): %s\n", k, length(members), join(members, ", "))
end
println("\nRJMCMC:")
for k in sort(unique(z_rjmcmc))
    members = df_clean.Country[z_rjmcmc .== k]
    @printf("  Cluster %d (%d members): %s\n", k, length(members), join(members, ", "))
end

p5 = scatter(pc1, pc2; group=z_marg,
    title="PC1 vs PC2 — Gibbs MAP clusters",
    xlabel="PC1", ylabel="PC2", legend=:outertopright)

p6 = scatter(pc1, pc2; group=z_rjmcmc,
    title="PC1 vs PC2 — RJMCMC MAP clusters",
    xlabel="PC1", ylabel="PC2", legend=:outertopright)

display(plot(p5, p6, layout=(1, 2), size=(1000, 450)))