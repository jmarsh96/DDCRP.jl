```@meta
CurrentModule = DistanceDependentCRP
```

# API Reference

Complete listing of all exported types and functions.

## Abstract Types

```@docs
LikelihoodModel
PoissonModel
BinomialModel
GammaModel
AbstractMCMCState
AbstractPriors
AbstractMCMCSamples
```

## Data Containers

```@docs
AbstractObservedData
CountData
CountDataWithTrials
CountDataWithPopulation
ContinuousData
observations
distance_matrix
trials
has_trials
population
has_population
get_missing_mask
has_missing
nobs
requires_trials
requires_population
```

## DDCRP Parameters and Options

```@docs
DDCRPParams
MCMCOptions
should_infer
get_prop_sd
```

## Birth Proposals

```@docs
BirthProposal
PriorProposal
ConjugateProposal
MomentMatchedProposal
NormalMomentMatch
InverseGammaMomentMatch
LogNormalMomentMatch
FixedDistributionProposal
MixedProposal
```

## Fixed-Dimension Proposals

```@docs
FixedDimensionProposal
NoUpdate
WeightedMean
Resample
MixedFixedDim
```

## Poisson Models

```@docs
PoissonClusterRates
PoissonClusterRatesState
PoissonClusterRatesPriors
PoissonClusterRatesSamples
PoissonClusterRatesMarg
PoissonClusterRatesMargState
PoissonClusterRatesMargPriors
PoissonClusterRatesMargSamples
PoissonPopulationRates
PoissonPopulationRatesState
PoissonPopulationRatesPriors
PoissonPopulationRatesSamples
PoissonPopulationRatesMarg
PoissonPopulationRatesMargState
PoissonPopulationRatesMargPriors
PoissonPopulationRatesMargSamples
```

## Binomial Models

```@docs
BinomialClusterProb
BinomialClusterProbState
BinomialClusterProbPriors
BinomialClusterProbSamples
BinomialClusterProbMarg
BinomialClusterProbMargState
BinomialClusterProbMargPriors
BinomialClusterProbMargSamples
```

## Gamma Models

```@docs
GammaClusterShapeMarg
GammaClusterShapeMargState
GammaClusterShapeMargPriors
GammaClusterShapeMargSamples
```

## Main MCMC Entry Point

```@docs
mcmc
```

## Model Interface Methods

```@docs
initialise_state
allocate_samples
extract_samples!
update_params!
table_contribution
posterior
update_c!
cluster_param_dicts
sample_birth_params
birth_params_logpdf
sample_birth_param
birth_param_logpdf
fixed_dim_params
fixed_dim_param
```

## DDCRP Core Utilities

```@docs
decay
construct_distance_matrix
simulate_ddcrp
precompute_log_ddcrp
ddcrp_contribution
compute_table_assignments
table_assignments_to_vector
table_vector
table_vector_minus_i
get_cluster_labels
c_to_z
```

## DDCRP Hyperparameter Sampling

```@docs
compute_R
count_self_links
sample_V!
update_α_ddcrp
update_s_ddcrp
update_s_ddcrp_augmented
```

## Diagnostics

```@docs
MCMCDiagnostics
MCMCSummary
record_move!
record_pairwise!
finalize!
acceptance_rates
pairwise_acceptance_rates
autocorrelation
integrated_autocorrelation_time
effective_sample_size
ess_per_second
summarize_mcmc
get_parameter_fields
compute_param_summary
```

## Simulation Utilities

```@docs
simulate_poisson_data
simulate_binomial_data
simulate_gamma_data
```

## Posterior Analysis

```@docs
calculate_n_clusters
posterior_num_cluster_distribution
compute_similarity_matrix
compute_ari_trace
compute_vi_trace
point_estimate_clustering
posterior_summary
compute_waic
compute_lpml
compute_psis_loo
posterior_predictive
```

## Sampler Internals

```@docs
get_moving_set
find_table_for_customer
update_c_rjmcmc!
update_c_gibbs!
save_entries
restore_entries!
sorted_setdiff
sorted_merge
```

## Proposal Utilities

```@docs
fit_inverse_gamma_moments
update_cluster_rates!
update_cluster_probs!
update_α!
logbinomial
```

## Prior Types

```@docs
PoissonPriors
BinomialPriors
```
