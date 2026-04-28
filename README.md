# DistanceDependentCRP.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jmarsh96.github.io/DistanceDependentCRP.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jmarsh96.github.io/DistanceDependentCRP.jl/dev/)
[![Build Status](https://github.com/jmarsh96/DistanceDependentCRP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jmarsh96/DistanceDependentCRP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jmarsh96/DistanceDependentCRP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jmarsh96/DistanceDependentCRP.jl)

`DistanceDependentCRP.jl` is a Julia package for Bayesian nonparametric clustering using the **Distance Dependent Chinese Restaurant Process** (DDCRP). It places a distance-dependent prior over partitions — observations that are close together are more likely to share a cluster — and supports Poisson, Binomial, and Gamma likelihood families. Inference is available via conjugate Gibbs sampling (for marginalised models) and Reversible Jump MCMC (for models with explicit cluster parameters).

## Installation

```julia
] add DistanceDependentCRP
```

Requires Julia 1.10 or later.

## Quickstart

```julia
using DistanceDependentCRP, Random

# Simulate clustered count data
Random.seed!(42)
sim = simulate_poisson_data(50, [1.0, 5.0, 10.0]; α=0.5, scale=1.0)

# Package data
data = CountData(sim.y, sim.D)

# Set DDCRP hyperparameters and priors
ddcrp_params = DDCRPParams(0.5, 1.0)
priors = PoissonClusterRatesMargPriors(1.0, 1.0)

# Run MCMC (marginalised model — Gibbs sampling)
opts = MCMCOptions(n_samples=2000, verbose=true)
samples = mcmc(PoissonClusterRatesMarg(), data, ddcrp_params, priors, ConjugateProposal(); opts=opts)

# Posterior summaries
psm = compute_similarity_matrix(samples.c)
c_est = point_estimate_clustering(samples.c)
println("Estimated number of clusters: ", maximum(c_est))
```

## Documentation

Full documentation (including model descriptions, API reference, and an extensibility guide) is available at:

- **Stable**: https://jmarsh96.github.io/DistanceDependentCRP.jl/stable/
- **Dev**: https://jmarsh96.github.io/DistanceDependentCRP.jl/dev/

## Contributing

**Reporting issues**: Open a [GitHub Issue](https://github.com/jmarsh96/DistanceDependentCRP.jl/issues) with a minimal working example (MWE) that reproduces the problem. Please include your Julia version, package version, and the full error message.

**Submitting changes**: Fork the repository, make your changes on a new branch, and open a pull request against `main`. Please ensure the test suite passes (`julia --project=. -e "using Pkg; Pkg.test()"`) before submitting.

## Acknowledgements

This work was supported by a UKRI Future Leaders Fellowship [MR/X034992/1] and the University of Nottingham.

The original paper which introduced the distance dependent Chinese restaurant process can be found in [1].

## References

[1] Blei, D.M. and Frazier, P.I., 2011. Distance dependent Chinese restaurant processes. Journal of Machine Learning Research, 12(8).
