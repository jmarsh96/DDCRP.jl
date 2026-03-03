# ============================================================================
# Walk Free – Hyperparameter Sensitivity Analysis
# ============================================================================
#
# Holds α_ddcrp and s_ddcrp fixed at each grid point (no inference) and runs
# NBPopulationRatesMarg + ConjugateProposal for all combinations in parallel.
#
# Usage:
#   julia --project scripts/walk_free_hyperparameter_analysis.jl [--slurm] [--test]
#
# Options:
#   --slurm   Spawn workers via SlurmClusterManager (reads SLURM_NTASKS)
#   --test    Use reduced iteration counts for a quick smoke-test
# ============================================================================

const USE_SLURM = "--slurm" in ARGS || "-slurm" in ARGS
const TEST_RUN  = "--test"  in ARGS || "-test"  in ARGS

if TEST_RUN
    println("*** TEST RUN MODE — reduced iteration counts ***")
end

# ============================================================================
# Worker setup — must happen before @everywhere
# ============================================================================

using Distributed

if USE_SLURM
    using SlurmClusterManager
    addprocs(SlurmClusterManager(); exeflags="--project=$(Base.active_project())")
    println("SLURM: $(nworkers()) worker(s) added")
else
    addprocs(4; exeflags="--project=$(Base.active_project())")
end

# ============================================================================
# Load packages on all workers (including main process)
# ============================================================================

@everywhere using DDCRP, JLD2

# ============================================================================
# Helper functions and per-run logic — defined on all workers
# ============================================================================

@everywhere begin

    # Format a float for use in a filename (e.g. 1.0 → "1p0", 50.0 → "50p0")
    _fmt(x) = replace(string(Float64(x)), "." => "p")

    # Thin samples by retaining every `thin_factor`-th iteration
    function _thin_samples(samples::NBPopulationRatesMargSamples, thin_factor::Int)
        idx = 1:thin_factor:size(samples.c, 1)
        NBPopulationRatesMargSamples(
            samples.c[idx, :],
            samples.λ[idx, :],
            samples.r[idx],
            samples.logpost[idx],
            samples.α_ddcrp[idx],
            samples.s_ddcrp[idx],
        )
    end

    function run_chain(
        α_val, s_val,
        y, P, D,
        priors, opts,
        outdir, outdir_thinned,
        thin_factor,
    )
        ddcrp_params = DDCRPParams(α_val, s_val)   # fixed — no prior hyperparams
        samples, diag = mcmc(
            NBPopulationRatesMarg(), y, P, D,
            ddcrp_params, priors, ConjugateProposal();
            opts = opts,
        )
        fname = "alpha_$(_fmt(α_val))_scale_$(_fmt(s_val)).jld2"
        @save joinpath(outdir, fname) samples diag ddcrp_params priors
        samples_thinned = _thin_samples(samples, thin_factor)
        @save joinpath(outdir_thinned, fname) samples_thinned diag ddcrp_params priors
        return (α_val, s_val, diag.total_time)
    end

end

# ============================================================================
# 1. Load and preprocess data (same as walk_free_analysis.jl)
# ============================================================================

using DataFrames, Distances, Statistics, XLSX, Random, Printf

Random.seed!(2025)

println("Loading GSI data...")
data_path = joinpath("data", "2023-Global-Slavery-Index-Data.xlsx")
df = DataFrame(XLSX.readtable(data_path, "GSI 2023 summary data"; first_row = 3))
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

# ============================================================================
# 2. Configuration
# ============================================================================

# NB population priors: γ ~ Gamma(2, rate=0.1), r ~ Gamma(1, rate=0.1)
const priors_hyp = NBPopulationRatesMargPriors(2.0, 0.1, 1.0, 0.1)

# Save every thin_factor-th iteration to the thinned output directory
const thin_factor = 10

if TEST_RUN
    n_samples = 2_000
else
    n_samples = 125_000
end

# α_ddcrp and s_ddcrp are NOT in infer_params — they are held fixed.
# DDCRPParams(α, s) sets the prior hyperparams to nothing, which also
# prevents inference even if the keys were present.
opts = MCMCOptions(
    n_samples         = n_samples,
    verbose           = false,
    track_diagnostics = true,
    infer_params      = Dict(:r => true, :c => true),
    prop_sds          = Dict(:λ => 0.3, :r => 0.5),
)

# Hyperparameter grid
# s is the decay scale: exp(-d*s). With standardised 5D Euclidean distances
# (typical range ~1–5), the decay half-life is log(2)/s.
# Small s (0.1–0.5) → long range, countries can link across larger distances.
# Large s (5–50)    → short range, only very close countries link.
α_values = 0.5:0.5:10.0
s_values = 0.5:0.5:5.0
combinations = vec([(α, s) for α in α_values, s in s_values])

outdir         = "results/walkfree_hyperparameters"
outdir_thinned = "results/walkfree_hyperparameters_thinned"
mkpath(outdir)
mkpath(outdir_thinned)

println("\nRunning $(length(combinations)) chains " *
        "($(length(α_values)) α × $(length(s_values)) s values)")
println("Workers:     $(nworkers())")
println("Samples:     $n_samples  | thin_factor: $thin_factor")
println("Outputs:     $outdir/")
println("             $outdir_thinned/\n")

# ============================================================================
# 3. Parallel execution via pmap
# ============================================================================

results = pmap(combinations) do (α_val, s_val)
    run_chain(α_val, s_val, y, P, D, priors_hyp, opts,
              outdir, outdir_thinned, thin_factor)
end

# ============================================================================
# 4. Summary table
# ============================================================================

println("\nAll chains completed:")
println(@sprintf("  %-10s  %-10s  %s", "α_ddcrp", "s_ddcrp", "Time (s)"))
println("  " * "-"^36)
for (α_val, s_val, t) in sort(results)
    println(@sprintf("  %-10.2g  %-10.2g  %.1f", α_val, s_val, t))
end

total_time = sum(r[3] for r in results)
println("\nTotal chain time (sum): $(round(total_time, digits=1)) s")
println("Files saved to:")
println("  Full samples:    $outdir/")
println("  Thinned samples: $outdir_thinned/")
