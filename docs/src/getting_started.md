```@meta
CurrentModule = DDCRP
```

# Getting Started

## Installation

`DDCRP.jl` requires Julia 1.10 or later. Install from the Julia REPL:

```julia
] add DDCRP
```

To use the package after installation:

```julia
using DDCRP
```

## Package Structure

The package is organised into five layers:

| Directory | Contents |
|---|---|
| `src/core/` | Abstract type hierarchy, data containers, DDCRP utilities, MCMC options |
| `src/models/` | Likelihood model implementations (`poisson/`, `binomial/`, `gamma/`) |
| `src/inference/` | Main `mcmc` loop, diagnostics, hyperparameter samplers, birth proposals |
| `src/samplers/` | Low-level Gibbs and RJMCMC move implementations |
| `src/utils/` | Data simulation and posterior analysis utilities |

## Choosing a Model

**By data type:**

- Integer counts with no exposure → `PoissonClusterRates` or `PoissonClusterRatesMarg`
- Integer counts with a known population/exposure per observation → `PoissonPopulationRates` or `PoissonPopulationRatesMarg`
- Successes out of a known number of trials → `BinomialClusterProb` or `BinomialClusterProbMarg`
- Positive continuous values → `GammaClusterShapeMarg`

**Marginalised vs. non-marginalised:**

- **Marginalised** (`*Marg`): cluster parameters are integrated out analytically. Faster, better mixing, but posterior samples of parameters are not available.
- **Non-marginalised**: explicit cluster parameters are sampled via RJMCMC. Use when you need posterior distributions over cluster rates/probabilities/shapes.

## Data Containers

**Constructors:**

```julia
# Poisson or simple Binomial
data = CountData(y, D)

# Binomial with N trials per observation (scalar or vector)
data = CountDataWithTrials(y, N, D)

# Poisson with population offsets; observation i has rate ρ_k × P_i
data = CountDataWithPopulation(y, P, D)                    # no missing data
data = CountDataWithPopulation(y, P, D, missing_mask)      # with missing mask

# Continuous (Gamma)
data = ContinuousData(y, D)
```

The distance matrix `D` is an *n × n* matrix of pairwise distances. To build one from a covariate vector `x`:

```julia
D = construct_distance_matrix(x)
```

## DDCRP Hyperparameters

`DDCRPParams` holds the concentration *α*, the decay scale *s*, and optionally Gamma hyperpriors for inference over *α* and *s*:

```julia
# Fixed hyperparameters
ddcrp = DDCRPParams(0.5, 1.0)

# Infer α, fix scale
ddcrp = DDCRPParams(0.5, 1.0, 1.0, 1.0)          # α_a=1, α_b=1

# Infer both α and scale
ddcrp = DDCRPParams(0.5, 1.0, 1.0, 1.0, 1.0, 1.0) # + s_a=1, s_b=1
```

## MCMC Options

Key options:

```julia
opts = MCMCOptions(
    n_samples     = 5000,
    verbose       = true,
    infer_params  = Dict(:α_ddcrp => true, :s_ddcrp => true),
    prop_sds      = Dict(:λ => 0.3, :s_ddcrp => 0.2),
    track_diagnostics = true,
)
```

Use `should_infer(opts, :param)` and `get_prop_sd(opts, :param)` when implementing custom models.

## Birth and Fixed-Dimension Proposals

Proposals control how RJMCMC birth moves sample new cluster parameters. Pass a `BirthProposal` as the fifth argument to `mcmc`.

| Proposal | Description |
|---|---|
| `ConjugateProposal()` | Triggers Gibbs sampling; use with all `*Marg` models |
| `PriorProposal()` | Sample new parameters from the prior |
| `LogNormalMomentMatch(σ)` | Log-normal proposal centred at empirical mean of moving set |
| `NormalMomentMatch(σ)` | Normal proposal centred at empirical mean |
| `InverseGammaMomentMatch()` | Moment-matched InverseGamma proposal |
| `FixedDistributionProposal(d)` | User-specified distribution |
| `MixedProposal(λ=p1, α=p2)` | Per-parameter proposals |

The `fixed_dim_proposal` keyword argument controls parameter updates for fixed-dimension RJMCMC moves (when K does not change):

| Fixed-dim proposal | Description |
|---|---|
| `NoUpdate()` | Keep parameters unchanged (default) |
| `WeightedMean()` | Deterministic weighted average update |
| `Resample(proposal)` | Stochastic resample using inner birth proposal |
| `MixedFixedDim(λ=p1)` | Per-parameter strategies |

## Running MCMC

**Marginalised (Gibbs) example:**

```julia
using DDCRP, Random
Random.seed!(1)

sim  = simulate_poisson_data(60, [2.0, 8.0]; α=0.5, scale=1.0)
data = CountData(sim.y, sim.D)

ddcrp  = DDCRPParams(0.5, 1.0)
priors = PoissonClusterRatesMargPriors(1.0, 0.5)
opts   = MCMCOptions(n_samples=3000)

samples = mcmc(PoissonClusterRatesMarg(), data, ddcrp, priors, ConjugateProposal(); opts=opts)
```

**Non-marginalised (RJMCMC) example:**

```julia
priors   = PoissonClusterRatesPriors(1.0, 0.5)
proposal = LogNormalMomentMatch(0.5)
fdim     = WeightedMean()

samples, diag = mcmc(
    PoissonClusterRates(), data, ddcrp, priors, proposal;
    fixed_dim_proposal=fdim, opts=opts
)

println(acceptance_rates(diag))
```

## Post-Processing

**Clustering summaries:**

```julia
# Posterior similarity matrix
psm = compute_similarity_matrix(samples.c)

# Point estimate via MAP
c_est = point_estimate_clustering(samples.c)

# Number of clusters per sample
nk = calculate_n_clusters(samples.c)

# Adjusted Rand Index vs. ground truth
ari = compute_ari_trace(samples.c, sim.c)
```

**MCMC diagnostics:**

```julia
summary = summarize_mcmc(samples, diag)

ess = effective_sample_size(samples.logpost)

rates = acceptance_rates(diag)   # birth/death/fixed-dimension accept rates
```

**Model comparison:**

```julia
# Requires a log-likelihood matrix (n_samples × n_obs)
waic = compute_waic(ll_matrix)
lpml = compute_lpml(ll_matrix)
loo  = compute_psis_loo(ll_matrix)
```

## Full End-to-End Example

```julia
using DDCRP, Random, Statistics

Random.seed!(42)

# 1. Simulate data
sim  = simulate_poisson_data(80, [1.0, 6.0, 15.0]; α=0.3, scale=2.0)
data = CountData(sim.y, sim.D)

# 2. Configure sampler
ddcrp  = DDCRPParams(0.3, 2.0)
priors = PoissonClusterRatesMargPriors(1.0, 0.5)
opts   = MCMCOptions(n_samples=5000, verbose=false)

# 3. Run MCMC
samples = mcmc(PoissonClusterRatesMarg(), data, ddcrp, priors, ConjugateProposal(); opts=opts)

# 4. Posterior summaries
psm   = compute_similarity_matrix(samples.c)
c_est = point_estimate_clustering(samples.c)
ari   = compute_ari_trace(samples.c, sim.c)

println("Mean ARI: ", round(mean(ari), digits=3))
println("Clusters found: ", maximum(c_est))
println("True clusters:  ", length(unique(sim.c)))
```
