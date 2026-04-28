# ============================================================================
# DDCRP.jl - Distance Dependent Chinese Restaurant Process
# Bayesian Nonparametric Clustering with Multiple Likelihood Models
# ============================================================================

module DDCRP

using Distributions
using SpecialFunctions
using Statistics
using StatsBase
using Random
using LinearAlgebra
using Clustering

# ============================================================================
# Exports
# ============================================================================

# Abstract types
export LikelihoodModel, AbstractMCMCState, BirthProposal
export PoissonModel, BinomialModel, GammaModel  # Abstract model families
export AbstractPriors, AbstractMCMCSamples

# Observed data types
export AbstractObservedData, CountData, CountDataWithTrials, CountDataWithPopulation
export observations, distance_matrix, trials, has_trials, nobs, requires_trials
export population, has_population, requires_population
export get_missing_mask, has_missing

# Birth proposals for RJMCMC
export PriorProposal, ConjugateProposal, MomentMatchedProposal
export NormalMomentMatch, InverseGammaMomentMatch, LogNormalMomentMatch
export FixedDistributionProposal, MixedProposal

# Fixed-dimension proposals for RJMCMC
export FixedDimensionProposal, NoUpdate, WeightedMean, Resample, MixedFixedDim

# Poisson model variants
export PoissonClusterRates, PoissonClusterRatesState, PoissonClusterRatesPriors, PoissonClusterRatesSamples
export PoissonClusterRatesMarg, PoissonClusterRatesMargState, PoissonClusterRatesMargPriors, PoissonClusterRatesMargSamples
export PoissonPopulationRates, PoissonPopulationRatesState, PoissonPopulationRatesPriors, PoissonPopulationRatesSamples
export PoissonPopulationRatesMarg, PoissonPopulationRatesMargState, PoissonPopulationRatesMargPriors, PoissonPopulationRatesMargSamples

# Binomial model variants
export BinomialClusterProb, BinomialClusterProbState, BinomialClusterProbPriors, BinomialClusterProbSamples
export BinomialClusterProbMarg, BinomialClusterProbMargState, BinomialClusterProbMargPriors, BinomialClusterProbMargSamples

# Gamma model variants
export GammaClusterShapeMarg
export GammaClusterShapeMargState, GammaClusterShapeMargPriors, GammaClusterShapeMargSamples
export ContinuousData
export update_α!

# DDCRP parameters
export DDCRPParams

# MCMC
export MCMCOptions, mcmc
export MCMCDiagnostics, MCMCSummary
export should_infer, get_prop_sd

# Interface methods
export table_contribution, posterior
export update_params!, update_cluster_rates!, update_cluster_probs!
export update_c!
export initialise_state, extract_samples!, allocate_samples

# RJMCMC interface methods
export cluster_param_dicts
export sample_birth_params, birth_params_logpdf, fixed_dim_params
export sample_birth_param, birth_param_logpdf

# Diagnostics
export acceptance_rates, pairwise_acceptance_rates
export effective_sample_size, integrated_autocorrelation_time, ess_per_second
export autocorrelation, summarize_mcmc
export record_move!, record_pairwise!, finalize!
export get_parameter_fields, compute_param_summary

# DDCRP hyperparameter sampling
export compute_R, count_self_links, sample_V!
export update_α_ddcrp, update_s_ddcrp, update_s_ddcrp_augmented

# Core DDCRP utilities
export construct_distance_matrix, decay
export precompute_log_ddcrp, ddcrp_contribution
export table_vector, table_vector_minus_i
export compute_table_assignments, table_assignments_to_vector
export simulate_ddcrp
export get_cluster_labels, c_to_z

export logbinomial

# Simulation utilities
export simulate_poisson_data, simulate_binomial_data, simulate_gamma_data

# Posterior predictive
export posterior_predictive

# Analysis utilities
export calculate_n_clusters, posterior_num_cluster_distribution
export compute_similarity_matrix, compute_ari_trace, compute_vi_trace
export compute_waic, compute_lpml, compute_psis_loo
export point_estimate_clustering, posterior_summary

# Sampler utilities
export get_moving_set, find_table_for_customer
export compute_fixed_dim_means, compute_weighted_means, resample_posterior_means
export update_c_rjmcmc!, save_entries, restore_entries!
export sorted_setdiff, sorted_merge

# Proposal utilities
export fit_inverse_gamma_moments, fit_gamma_shape_moments

# ============================================================================
# Include files in dependency order
# ============================================================================

# Core types and infrastructure (no dependencies)
include("core/types.jl")
include("core/priors.jl")
include("core/ddcrp.jl")
include("core/options.jl")
include("core/rjmcmc_interface.jl")

# Inference machinery - proposals and diagnostics (depends on core)
include("inference/proposals.jl")
include("inference/diagnostics.jl")
include("inference/hyperparams.jl")

# Model implementations - Poisson variants
include("models/poisson/poisson_cluster_rates.jl")
include("models/poisson/poisson_cluster_rates_marg.jl")
include("models/poisson/poisson_population_rates.jl")
include("models/poisson/poisson_population_rates_marg.jl")

# Model implementations - Binomial variants
include("models/binomial/binomial_utils.jl")
include("models/binomial/binomial_cluster_prob.jl")
include("models/binomial/binomial_cluster_prob_marg.jl")

# Model implementations - Gamma variants
include("models/gamma/gamma_cluster_shape_marg.jl")

# Samplers (depends on model types for specialized methods)
include("samplers/gibbs.jl")
include("samplers/rjmcmc.jl")

# MCMC loop (depends on models and samplers)
include("inference/mcmc.jl")

# Utilities (depends on core)
include("utils/simulation.jl")
include("utils/analysis.jl")

end # module
