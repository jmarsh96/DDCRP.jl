```@meta
CurrentModule = DDCRP
```

# DDCRP.jl

`DDCRP.jl` is a Julia package for Bayesian nonparametric clustering using the **Distance Dependent Chinese Restaurant Process** (DDCRP).

## What is the DDCRP?

The standard Chinese Restaurant Process (CRP) places a prior over partitions of *n* observations, favouring partitions with few, large clusters. The DDCRP extends this by making cluster membership depend on the **distances** between observations: each customer *i* links to another customer *j* (or to themselves) with probability proportional to a decay function *f(d_{ij})* of their distance. When customer *i* links to customer *j*, *i* inherits *j*'s cluster. The resulting partitions are determined by the **connected components** of the directed customer–link graph.

The default decay function is exponential:

```
f(d; scale) = exp(-d × scale)
```

A self-link (`c[i] = i`) acts as a table head — the customer starts their own cluster. The concentration parameter *α* controls the prior probability of self-linking: higher *α* produces more clusters.

## Available Models

| Model | Likelihood | Parameters | Inference |
|---|---|---|---|
| `PoissonClusterRates` | Poisson | Cluster rates λ_k | RJMCMC |
| `PoissonClusterRatesMarg` | Poisson | (marginalised) | Gibbs |
| `PoissonPopulationRates` | Poisson + exposure | Cluster rates ρ_k | RJMCMC |
| `PoissonPopulationRatesMarg` | Poisson + exposure | (marginalised) | Gibbs |
| `BinomialClusterProb` | Binomial | Cluster probabilities p_k | RJMCMC |
| `BinomialClusterProbMarg` | Binomial | (marginalised) | Gibbs |
| `GammaClusterShapeMarg` | Gamma | Cluster shapes α_k | RJMCMC |

**Marginalised** variants integrate out cluster parameters analytically and use Gibbs sampling — they are faster and mix better but do not provide posterior samples of the cluster parameters themselves. **Non-marginalised** variants carry explicit cluster parameters and use Reversible Jump MCMC (RJMCMC).

## Navigation

- [Getting Started](getting_started.md) — installation, data setup, running MCMC, post-processing
- [Poisson Models](models/poisson.md) — count data likelihood models
- [Binomial Models](models/binomial.md) — success/trial likelihood models
- [Gamma Models](models/gamma.md) — positive continuous likelihood models
- [Adding Your Own Model](extending.md) — implementing new likelihood families
- [API Reference](api.md) — complete type and function reference
