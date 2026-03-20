# ============================================================================
# Walk Free – NBPopulationRates birth/fixed-dim proposal comparison
# ============================================================================
#
# Runs the RJMCMC sampler (NBPopulationRates) with every combination of
# birth proposal and fixed-dimension proposal, plus one Gibbs reference run
# (NBPopulationRatesMarg). All chains are saved to results/walkfree/chains/.
# A summary CSV is written when all jobs complete.
#
# CLI options:
#   --test         reduced iteration count (smoke test)
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

Random.seed!(42)

# ── Output directories ─────────────────────────────────────────────────────────
mkpath("results/walkfree/chains")

# ============================================================================
# Data loading (main process; passed to workers as function arguments)
# ============================================================================

println("Loading GSI data...")
data_path = joinpath("data", "GSI_with_survey.xlsx")
df = DataFrame(XLSX.readtable(data_path, "GSI 2023 summary data"))
rename!(df, Dict(
    "Estimated number of people in modern slavery" => :est_num_ms,
    "Population"              => :pop,
    "Governance issues"       => :governance_issues,
    "Lack of basic needs"     => :lack_basic_needs,
    "Inequality"              => :inequality,
    "Disenfranchised groups"  => :disenfranchised_groups,
    "Effects of conflict"     => :effects_of_conflict,
))
filter!(x -> x.pop > 0, df)

covariate_cols = [
    :governance_issues, :lack_basic_needs, :inequality,
    :disenfranchised_groups, :effects_of_conflict,
]
required_cols = [:est_num_ms, :pop, covariate_cols...]
df_clean = dropmissing(df, required_cols)

# y: missing for unsurveyed countries
y_raw = ifelse.(df_clean.Surveyed, df_clean.est_num_ms, missing)
P     = Int.(df_clean.pop)

X     = Float64.(Matrix(df_clean[:, covariate_cols]))
X_std = (X .- mean(X, dims=1)) ./ std(X, dims=1)
D     = pairwise(Euclidean(), X_std', dims=2)
n     = length(y_raw)

n_obs     = n
n_missing = sum(ismissing.(y_raw))
println("  n=$n_obs observations, $n_missing missing")

# ── Shared MCMC settings ───────────────────────────────────────────────────────
const N_SAMPLES_FULL = 250_000
const N_SAMPLES_TEST = 2_000
n_samples = TEST_RUN ? N_SAMPLES_TEST : N_SAMPLES_FULL
burnin    = div(n_samples, 4)

γ_a = 1.0; γ_b = 0.1
r_a = 1.0; r_b = 0.1

ddcrp_params  = DDCRPParams(1.0, 1.0, 1.0, 0.1, 1.0, 0.1)
priors_marg   = NBPopulationRatesMargPriors(γ_a, γ_b, r_a, r_b)
priors_unmarg = NBPopulationRatesPriors(γ_a, γ_b, r_a, r_b)

opts_common = MCMCOptions(
    n_samples         = n_samples,
    verbose           = false,
    track_diagnostics = true,
    infer_params      = Dict(
        :s_ddcrp => false,
        :r       => true,
        :α_ddcrp => true,
    ),
)

# ============================================================================
# Worker function (defined @everywhere)
# ============================================================================

@everywhere begin

"""
Reconstruct a birth proposal from a serialisable description.
  cfg.birth        ∈ {:prior, :fixed_dist, :lognormal, :invgamma}
  cfg.birth_params — NamedTuple with θ (fixed_dist), σ (lognormal), min_size (invgamma)
"""
function _make_birth(cfg)
    b = cfg.birth
    p = cfg.birth_params
    if b == :prior
        return PriorProposal()
    elseif b == :fixed_dist
        return FixedDistributionProposal([Exponential(p.θ)])
    elseif b == :lognormal
        return LogNormalMomentMatch([p.σ])
    elseif b == :invgamma
        return InverseGammaMomentMatch(p.min_size)
    else
        error("Unknown birth proposal: $b")
    end
end

"""
Reconstruct a fixed-dim proposal from a serialisable description.
  cfg.fixed_dim ∈ {:noupdate, :weighted_mean, :resample}
"""
function _make_fixed_dim(cfg)
    fd = cfg.fixed_dim
    if fd == :noupdate
        return NoUpdate()
    elseif fd == :weighted_mean
        return WeightedMean()
    elseif fd == :resample
        return Resample(_make_birth(cfg))
    else
        error("Unknown fixed-dim proposal: $fd")
    end
end

"""
Run one RJMCMC configuration. `cfg` is a serialisable NamedTuple; all
data and options are passed explicitly so the function works on remote workers.
"""
function run_rjmcmc_config(cfg, y, P, D, ddcrp_params, priors_unmarg, opts)
    birth_prop    = _make_birth(cfg)
    fixed_dim_prop = _make_fixed_dim(cfg)

    t0 = time()
    samples, diag = mcmc(
        NBPopulationRates(), y, P, D, ddcrp_params, priors_unmarg, birth_prop;
        fixed_dim_proposal = fixed_dim_prop,
        opts               = opts,
        init_params        = Dict{Symbol,Any}(:r => 10.0),
    )
    elapsed = time() - t0

    out_path = "results/walkfree/chains/$(cfg.name).jld2"
    jldsave(out_path; samples, diag, cfg)

    n_iter   = size(samples.c, 1)
    idx      = (div(n_iter, 4) + 1):n_iter
    nc       = calculate_n_clusters(samples.c[idx, :])
    ar       = acceptance_rates(diag)

    return (
        name          = cfg.name,
        mean_K        = mean(nc),
        median_K      = Int(median(nc)),
        mode_K        = mode(nc),
        std_K         = std(nc),
        ess_K         = effective_sample_size(Float64.(nc)),
        mean_r        = mean(samples.r[idx]),
        mean_α        = mean(samples.α_ddcrp[idx]),
        ar_birth      = ar.birth,
        ar_death      = ar.death,
        ar_fixed      = ar.fixed,
        time_s        = elapsed,
    )
end

end # @everywhere

# ============================================================================
# Build the configuration grid
# ============================================================================

# Birth proposals: (symbol, params NamedTuple, label suffix)
birth_specs = [
    (:prior,      (θ=0.0,  σ=0.0, min_size=0),  "prior"),
    (:fixed_dist, (θ=1.0,  σ=0.0, min_size=0),  "fixeddist_exp1"),
    (:fixed_dist, (θ=5.0,  σ=0.0, min_size=0),  "fixeddist_exp5"),
    (:fixed_dist, (θ=10.0, σ=0.0, min_size=0),  "fixeddist_exp10"),
    (:lognormal,  (θ=0.0,  σ=0.3, min_size=0),  "lognormal_s03"),
    (:lognormal,  (θ=0.0,  σ=0.5, min_size=0),  "lognormal_s05"),
    (:lognormal,  (θ=0.0,  σ=1.0, min_size=0),  "lognormal_s10"),
    (:invgamma,   (θ=0.0,  σ=0.0, min_size=2),  "invgamma_k2"),
    (:invgamma,   (θ=0.0,  σ=0.0, min_size=5),  "invgamma_k5"),
    (:invgamma,   (θ=0.0,  σ=0.0, min_size=10), "invgamma_k10"),
]

# Fixed-dim proposals: (symbol, label suffix)
fixed_dim_specs = [
    (:noupdate,     "noupdate"),
    (:weighted_mean, "weighted"),
    (:resample,     "resample"),
]

rjmcmc_configs = NamedTuple[]
for (bsym, bparams, blabel) in birth_specs
    for (fdsym, fdlabel) in fixed_dim_specs
        push!(rjmcmc_configs, (
            name         = "rjmcmc_$(blabel)_$(fdlabel)",
            birth        = bsym,
            birth_params = bparams,
            fixed_dim    = fdsym,
        ))
    end
end

println("\nProposal grid: $(length(rjmcmc_configs)) RJMCMC configurations")
for cfg in rjmcmc_configs
    println("  $(cfg.name)")
end

# ============================================================================
# Step 1 – Gibbs reference (NBPopulationRatesMarg, main process)
# ============================================================================

println("\n=== Gibbs reference (NBPopulationRatesMarg) ===")
t0_gibbs = time()
samples_marg, diag_marg = mcmc(
    NBPopulationRatesMarg(), y_raw, P, D, ddcrp_params, priors_marg, ConjugateProposal();
    opts        = opts_common,
    init_params = Dict{Symbol,Any}(:r => 10.0),
)
elapsed_gibbs = time() - t0_gibbs

jldsave("results/walkfree/chains/gibbs_marg.jld2"; samples=samples_marg, diag=diag_marg)

idx_g   = (burnin + 1):n_samples
nc_g    = calculate_n_clusters(samples_marg.c[idx_g, :])
gibbs_summary = (
    name     = "gibbs_marg",
    mean_K   = mean(nc_g),
    median_K = Int(median(nc_g)),
    mode_K   = mode(nc_g),
    std_K    = std(nc_g),
    ess_K    = effective_sample_size(Float64.(nc_g)),
    mean_r   = mean(samples_marg.r[idx_g]),
    mean_α   = mean(samples_marg.α_ddcrp[idx_g]),
    ar_birth = NaN,
    ar_death = NaN,
    ar_fixed = NaN,
    time_s   = elapsed_gibbs,
)

@printf("  mean_K=%.2f  mode_K=%d  ESS(K)=%.1f  time=%.1fs\n",
        gibbs_summary.mean_K, gibbs_summary.mode_K, gibbs_summary.ess_K, elapsed_gibbs)

# ============================================================================
# Step 2 – RJMCMC grid via pmap
# ============================================================================

println("\n=== RJMCMC proposal comparison ($(length(rjmcmc_configs)) configs via pmap) ===")

rjmcmc_results = pmap(rjmcmc_configs) do cfg
    run_rjmcmc_config(cfg, y_raw, P, D, ddcrp_params, priors_unmarg, opts_common)
end

# ============================================================================
# Step 3 – Summary table
# ============================================================================

all_results = vcat([gibbs_summary], rjmcmc_results)

println("\n\n--- Posterior summary (post-burnin) ---")
@printf("%-40s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n",
        "Config", "mean_K", "mode_K", "std_K", "ESS_K",
        "mean_r", "mean_α", "ar_birth", "ar_death", "time_s")
println("-"^120)
for r in all_results
    @printf("%-40s %8.2f %8d %8.2f %8.1f %8.4f %8.4f %8.4f %8.4f %8.1f\n",
            r.name, r.mean_K, r.mode_K, r.std_K, r.ess_K,
            r.mean_r, r.mean_α,
            isnan(r.ar_birth) ? 0.0 : r.ar_birth,
            isnan(r.ar_death) ? 0.0 : r.ar_death,
            r.time_s)
end

df_summary = DataFrame(
    config   = [r.name    for r in all_results],
    mean_K   = [r.mean_K  for r in all_results],
    median_K = [r.median_K for r in all_results],
    mode_K   = [r.mode_K  for r in all_results],
    std_K    = [r.std_K   for r in all_results],
    ess_K    = [r.ess_K   for r in all_results],
    mean_r   = [r.mean_r  for r in all_results],
    mean_α   = [r.mean_α  for r in all_results],
    ar_birth = [r.ar_birth for r in all_results],
    ar_death = [r.ar_death for r in all_results],
    ar_fixed = [r.ar_fixed for r in all_results],
    time_s   = [r.time_s  for r in all_results],
)

csv_path = "results/walkfree/proposal_comparison.csv"
CSV.write(csv_path, df_summary)
println("\nSummary saved to $csv_path")
println("Chains saved to results/walkfree/chains/")
