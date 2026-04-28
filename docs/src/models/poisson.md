```@meta
CurrentModule = DistanceDependentCRP
```

# Poisson Models

Poisson models are appropriate for non-negative integer count data. Each cluster *k* has a rate parameter *λ_k*, and the likelihood for observation *i* in cluster *k* is:

```
y_i | λ_k ~ Poisson(λ_k)
```

For population models the rate is *ρ_k × P_i*, where *P_i* is a known exposure or population size:

```
y_i | ρ_k, P_i ~ Poisson(ρ_k × P_i)
```

All Poisson models use a conjugate Gamma prior on the rate, which allows analytical marginalisation.

---

## PoissonClusterRates

Explicit cluster rates *λ_k* sampled via RJMCMC. Use this when you need posterior distributions over the cluster rates themselves. State holds a `λ_dict::Dict{Vector{Int},Float64}` mapping sorted customer index vectors to rates; priors are `λ_a, λ_b` for the `Gamma(λ_a, λ_b)` prior. See the [API Reference](../api.md#Poisson-Models) for full type documentation.

**Example:**

```julia
data = CountData(y, D)
ddcrp = DDCRPParams(0.5, 1.0)
priors = PoissonClusterRatesPriors(1.0, 0.5)
opts = MCMCOptions(n_samples=3000)

samples, diag = mcmc(
    PoissonClusterRates(), data, ddcrp, priors, LogNormalMomentMatch(0.5);
    fixed_dim_proposal=WeightedMean(), opts=opts
)
```

---

## PoissonClusterRatesMarg

Cluster rates *λ_k* are integrated out analytically. Inference uses conjugate Gibbs sampling, which is faster and typically mixes better than RJMCMC. State holds only `c::Vector{Int}`; no parameter dictionaries. See the [API Reference](../api.md#Poisson-Models) for full type documentation.

**Example:**

```julia
data = CountData(y, D)
priors = PoissonClusterRatesMargPriors(1.0, 0.5)

samples = mcmc(PoissonClusterRatesMarg(), data, ddcrp, priors, ConjugateProposal(); opts=opts)
```

---

## PoissonPopulationRates

Cluster rate multipliers *ρ_k* with per-observation population offsets *P_i*. The expected count for observation *i* is *ρ_k × P_i*. Supports missing observations via a `BitVector` mask on `CountDataWithPopulation`. See the [API Reference](../api.md#Poisson-Models) for full type documentation.

**Example:**

```julia
data = CountDataWithPopulation(y, population, D)
priors = PoissonPopulationRatesPriors(1.0, 0.5)

samples, diag = mcmc(
    PoissonPopulationRates(), data, ddcrp, priors, LogNormalMomentMatch(0.5);
    fixed_dim_proposal=WeightedMean(), opts=opts
)
```

**With missing data:**

```julia
mask = BitVector([false, true, false, ...])   # true = missing
data = CountDataWithPopulation(y, population, D, mask)
```

Missing observations contribute only through the DDCRP prior, not the likelihood.

---

## PoissonPopulationRatesMarg

Like `PoissonPopulationRates` but with cluster rate multipliers *ρ_k* marginalised out. Uses conjugate Gibbs sampling. See the [API Reference](../api.md#Poisson-Models) for full type documentation.

**Example:**

```julia
data = CountDataWithPopulation(y, population, D)
priors = PoissonPopulationRatesMargPriors(1.0, 0.5)

samples = mcmc(PoissonPopulationRatesMarg(), data, ddcrp, priors, ConjugateProposal(); opts=opts)
```

---

## Simulation

See [`simulate_poisson_data`](@ref) in the [API Reference](../api.md#Simulation-Utilities).
