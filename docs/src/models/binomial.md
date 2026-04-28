```@meta
CurrentModule = DDCRP
```

# Binomial Models

Binomial models are appropriate for data consisting of successes *y_i* out of *N_i* trials. Each cluster *k* has a success probability *p_k*, and the likelihood for observation *i* in cluster *k* is:

```
y_i | p_k, N_i ~ Binomial(N_i, p_k)
```

The number of trials *N* can be a single integer shared across all observations (scalar) or a per-observation vector.

All Binomial models use a conjugate Beta prior on *p_k*, which allows analytical marginalisation.

---

## BinomialClusterProb

Explicit cluster probabilities *p_k* sampled via RJMCMC. Use this when you need posterior distributions over the cluster success probabilities themselves. State holds a `p_dict::Dict{Vector{Int},Float64}`; priors are `p_a, p_b` for `Beta(p_a, p_b)`. See the [API Reference](../api.md#Binomial-Models) for full type documentation.

**Example:**

```julia
# Scalar N: same number of trials for every observation
data = CountDataWithTrials(y, 10, D)

# Vector N: different number of trials per observation
data = CountDataWithTrials(y, N_vec, D)

ddcrp = DDCRPParams(0.5, 1.0)
priors = BinomialClusterProbPriors(1.0, 1.0)   # Beta(1,1) = Uniform prior
opts = MCMCOptions(n_samples=3000)

samples, diag = mcmc(
    BinomialClusterProb(), data, ddcrp, priors, LogNormalMomentMatch(0.5);
    fixed_dim_proposal=WeightedMean(), opts=opts
)
```

---

## BinomialClusterProbMarg

Cluster probabilities *p_k* are integrated out analytically via Beta–Binomial conjugacy. Inference uses conjugate Gibbs sampling. State holds only `c::Vector{Int}`. See the [API Reference](../api.md#Binomial-Models) for full type documentation.

**Example:**

```julia
data = CountDataWithTrials(y, N, D)
priors = BinomialClusterProbMargPriors(1.0, 1.0)

samples = mcmc(BinomialClusterProbMarg(), data, ddcrp, priors, ConjugateProposal(); opts=opts)
```

---

## Simulation

See [`simulate_binomial_data`](@ref) in the [API Reference](../api.md#Simulation-Utilities).
