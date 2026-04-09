# ============================================================================
# compare_poisson_population_marg_vs_unmarg.jl
#
# Sanity check: PoissonPopulationRatesMarg (marginalised) and
# PoissonPopulationRates (unmarginalised) should recover the same
# posterior distribution over the number of clusters K.
#
# Setup: 60 observations in 2 spatial clusters (30 each), well-separated
#        rates ρ = [0.5, 3.0], population P_i ~ Uniform(500, 3000).
# ============================================================================

using DDCRP, Random, Distributions, Statistics
using Printf, Plots, StatsPlots, StatsBase

Random.seed!(123)

# ============================================================================
# Simulate data
# ============================================================================

n_per_cluster = 30
n_clusters    = 2
n             = n_per_cluster * n_clusters
ρ_true        = [0.5, 3.0]
centres       = [0.0, 6.0]

z_true = repeat(1:n_clusters, inner=n_per_cluster)
X      = zeros(n)
for (k, cx) in enumerate(centres)
    idx = z_true .== k
    X[idx] = cx .+ randn(n_per_cluster)
end

D = [abs(X[i] - X[j]) for i in 1:n, j in 1:n]
P = rand(500:3000, n)
y = [rand(Poisson(P[i] * ρ_true[z_true[i]])) for i in 1:n]

println("True cluster counts: ", [sum(z_true .== k) for k in 1:n_clusters])
println("Mean y per cluster:  ", [round(mean(y[z_true .== k]); digits=1) for k in 1:n_clusters])

# ============================================================================
# Shared MCMC settings
# ============================================================================

ddcrp_params = DDCRPParams(1.0, 2.0)
ρ_a, ρ_b     = 1.0, 0.5          # same prior for both models
n_samples     = 250_000
burn          = div(n_samples, 5)

opts = MCMCOptions(
    n_samples   = n_samples,
    verbose     = true,
    infer_params = Dict(:α_ddcrp => true, :s_ddcrp => false),
)

# ============================================================================
# Run marginalised model
# ============================================================================

println("\n--- PoissonPopulationRatesMarg (marginalised) ---")
model_marg  = PoissonPopulationRatesMarg()
priors_marg = PoissonPopulationRatesMargPriors(ρ_a, ρ_b)

samples_marg, _ = mcmc(model_marg, y, P, D, ddcrp_params, priors_marg,
                        ConjugateProposal(); opts=opts)

# ============================================================================
# Run unmarginalised model
# ============================================================================

println("\n--- PoissonPopulationRates (unmarginalised) ---")
model_unmarg  = PoissonPopulationRates()
priors_unmarg = PoissonPopulationRatesPriors(ρ_a, ρ_b)

samples_unmarg, _ = mcmc(model_unmarg, y, P, D, ddcrp_params, priors_unmarg,
                          PriorProposal(); opts=opts)

# ============================================================================
# Compare posterior K distributions (post burn-in)
# ============================================================================

k_marg   = calculate_n_clusters(samples_marg.c[(burn+1):end, :])
k_unmarg = calculate_n_clusters(samples_unmarg.c[(burn+1):end, :])

k_all  = sort(unique([k_marg; k_unmarg]))
p_marg   = [mean(k_marg   .== k) for k in k_all]
p_unmarg = [mean(k_unmarg .== k) for k in k_all]

println("\n=== Posterior K distribution ===")
println(@sprintf("%-4s  %-10s  %-10s", "K", "Marg", "Unmarg"))
println(repeat("-", 30))
for (k, pm, pu) in zip(k_all, p_marg, p_unmarg)
    println(@sprintf("%-4d  %-10.4f  %-10.4f", k, pm, pu))
end

println("\nMarginalised  — mean K: $(round(mean(k_marg);   digits=2)), mode K: $(mode(k_marg))")
println("Unmarginalised — mean K: $(round(mean(k_unmarg); digits=2)), mode K: $(mode(k_unmarg))")

# ============================================================================
# Plots
# ============================================================================

# --- Posterior K bar chart ---
p1 = groupedbar(
    k_all,
    hcat(p_marg, p_unmarg);
    bar_position = :dodge,
    label        = ["Marginalised" "Unmarginalised"],
    xlabel       = "Number of clusters K",
    ylabel       = "Posterior probability",
    title        = "Posterior P(K) — marg vs unmarg\n(true K = $n_clusters)",
    color        = [:steelblue :darkorange],
    xticks       = k_all,
    legend       = :topright,
)
vline!(p1, [n_clusters]; color=:red, linestyle=:dash, linewidth=2, label="True K")

# --- Trace plots for K ---
p2 = plot(
    k_marg;
    label     = "Marginalised",
    xlabel    = "Iteration (post burn-in)",
    ylabel    = "K",
    title     = "K trace (post burn-in)",
    color     = :steelblue,
    alpha     = 0.7,
    linewidth = 0.5,
)
plot!(p2, k_unmarg; label="Unmarginalised", color=:darkorange, alpha=0.7, linewidth=0.5)
hline!(p2, [n_clusters]; color=:red, linestyle=:dash, linewidth=2, label="True K")

fig = plot(p1, p2; layout=(1, 2), size=(1100, 450), margin=8Plots.mm)
out_path = joinpath(@__DIR__, "compare_poisson_population.png")
savefig(fig, out_path)
println("\nPlot saved to $out_path")
display(fig)
