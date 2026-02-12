using DDCRP, Distributions, StatsBase, Random, BenchmarkTools

# ============================================================================
# Setup: same data as scripts/gamma.jl
# ============================================================================

Random.seed!(123)
n = 120
n_clusters = 3
cluster_x_means = [0.0, 5.0, 10.0]
cluster_x_std = 0.5
x = vcat([rand(Normal(mean, cluster_x_std), n ÷ n_clusters) for mean in cluster_x_means]...)
cluster_assignments = repeat(1:n_clusters, inner=n ÷ n_clusters)
D = construct_distance_matrix(x)

α_true = [1.0, 2.0, 3.0]
β_true = [1.0, 1.0, 1.0]
y = rand.(Gamma.(α_true[cluster_assignments], β_true[cluster_assignments]))

model = GammaClusterShapeMarg()
data = ContinuousData(y, D)
ddcrp_params = DDCRPParams(1.0, 1.0)
priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)
proposal = PriorProposal()
opts = MCMCOptions(n_samples=100, verbose=false)

log_DDCRP = DDCRP.precompute_log_ddcrp(
    ddcrp_params.decay_fn, ddcrp_params.α, ddcrp_params.scale, D)

# Create a well-initialized state for benchmarking
Random.seed!(42)
state_base = DDCRP.initialise_state(model, data, ddcrp_params, priors)

# ============================================================================
# Benchmark: Single sweep (n customer updates)
# ============================================================================

function bench_original_sweep(model, state, data, priors, proposal, log_DDCRP, opts)
    for i in 1:DDCRP.nobs(data)
        DDCRP.update_c_rjmcmc!(model, i, state, data, priors, proposal, log_DDCRP, opts)
    end
end

function bench_cached_sweep(model, state, data, priors, proposal, log_DDCRP, opts)
    for i in 1:DDCRP.nobs(data)
        DDCRP.update_c_rjmcmc_cached!(model, i, state, data, priors, proposal, log_DDCRP, opts)
    end
end

println("=" ^ 60)
println("Benchmark: Single sweep of n=$n customer updates")
println("=" ^ 60)

println("\n--- Original update_c_rjmcmc! ---")
b_orig = @benchmark bench_original_sweep(
    $model, state_copy, $data, $priors, $proposal, $log_DDCRP, $opts
) setup=(state_copy = deepcopy($state_base))
display(b_orig)

println("\n--- Cached update_c_rjmcmc_cached! ---")
b_cached = @benchmark bench_cached_sweep(
    $model, state_copy, $data, $priors, $proposal, $log_DDCRP, $opts
) setup=(state_copy = deepcopy($state_base))
display(b_cached)

println("\n" * "=" ^ 60)
println("Speedup Summary")
println("=" ^ 60)
speedup = median(b_orig).time / median(b_cached).time
println("Median time speedup: $(round(speedup, digits=2))x")
println("Allocations: $(b_orig.allocs) -> $(b_cached.allocs)")
println("Memory: $(b_orig.memory) bytes -> $(b_cached.memory) bytes")

# ============================================================================
# Benchmark: Full MCMC run
# ============================================================================

println("\n" * "=" ^ 60)
println("Benchmark: Full MCMC (1000 iterations)")
println("=" ^ 60)

full_opts = MCMCOptions(n_samples=1_000, verbose=false)

println("\n--- Original (via update_c_rjmcmc!) ---")
# Temporarily swap back to original for benchmarking
b_full_orig = @benchmark begin
    Random.seed!(seed)
    state = DDCRP.initialise_state($model, $data, $ddcrp_params, $priors)
    for i in 1:DDCRP.nobs($data)
        DDCRP.update_c_rjmcmc!(
            $model, i, state, $data, $priors, $proposal, $log_DDCRP, $opts)
    end
end setup=(seed = rand(UInt))
display(b_full_orig)

println("\n--- Cached (via update_c_rjmcmc_cached!) ---")
b_full_cached = @benchmark begin
    Random.seed!(seed)
    state = DDCRP.initialise_state($model, $data, $ddcrp_params, $priors)
    for i in 1:DDCRP.nobs($data)
        DDCRP.update_c_rjmcmc_cached!(
            $model, i, state, $data, $priors, $proposal, $log_DDCRP, $opts)
    end
end setup=(seed = rand(UInt))
display(b_full_cached)

full_speedup = median(b_full_orig).time / median(b_full_cached).time
println("\nFull MCMC median time speedup: $(round(full_speedup, digits=2))x")
println("Allocations: $(b_full_orig.allocs) -> $(b_full_cached.allocs)")
println("Memory: $(b_full_orig.memory) bytes -> $(b_full_cached.memory) bytes")



@code_typed DDCRP.update_c_rjmcmc!(model, 1, state_base, data, priors, proposal, log_DDCRP, opts)