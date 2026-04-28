```@meta
CurrentModule = DDCRP
```

# Gamma Models

Gamma models are appropriate for positive continuous data. Each cluster *k* has a shape parameter *α_k* and a rate parameter *β_k*, and the likelihood for observation *i* in cluster *k* is:

```
y_i | α_k, β_k ~ Gamma(α_k, β_k)
```

Currently one Gamma model is implemented: `GammaClusterShapeMarg`, which marginalises out the rate parameters *β_k* analytically and samples the shape parameters *α_k* via Metropolis–Hastings.

---

## GammaClusterShapeMarg

The rate *β_k* is integrated out analytically using a conjugate Gamma prior, while the shape *α_k* is sampled via Metropolis–Hastings with a log-normal random walk proposal. The proposal standard deviation for *α_k* is controlled by `prop_sds[:α]` in `MCMCOptions`. See the [API Reference](../api.md#Gamma-Models) for full type documentation.

**Example:**

```julia
using DDCRP, Random
Random.seed!(7)

sim = simulate_gamma_data(60, [2.0, 8.0], [1.0, 1.0]; α=0.3, scale=1.5)
data = ContinuousData(sim.y, sim.D)

ddcrp = DDCRPParams(0.3, 1.5)
priors = GammaClusterShapeMargPriors(α_a=2.0, α_b=1.0, β_a=2.0, β_b=1.0)
opts = MCMCOptions(n_samples=5000, prop_sds=Dict(:α => 0.3))

# GammaClusterShapeMarg uses RJMCMC (shape α is explicit)
samples, diag = mcmc(
    GammaClusterShapeMarg(), data, ddcrp, priors, LogNormalMomentMatch(0.3);
    fixed_dim_proposal=WeightedMean(), opts=opts
)

println(acceptance_rates(diag))
```

**Tuning tips:**

- The MH proposal SD for *α_k* is set via `prop_sds=Dict(:α => σ)`. Values around 0.2–0.5 work well for moderate-sized clusters.
- If acceptance rates are too low, reduce `prop_sds[:α]`; if too high (> 0.8), increase it.
- `GammaClusterShapeMarg` always uses RJMCMC because the shape parameter is not conjugately marginalised. Use `PriorProposal()`, `LogNormalMomentMatch(σ)`, or `FixedDistributionProposal(d)` as birth proposals.

---

## Simulation

See [`simulate_gamma_data`](@ref) in the [API Reference](../api.md#Simulation-Utilities).
