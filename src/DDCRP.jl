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
export NegativeBinomialModel, PoissonModel, BinomialModel  # Abstract model families
export AbstractPriors, AbstractMCMCSamples

# Observed data types
export AbstractObservedData, CountData, CountDataWithTrials
export observations, distance_matrix, trials, has_trials, nobs, requires_trials

# Birth proposals for RJMCMC
export PriorProposal, ConjugateProposal, MomentMatchedProposal
export NormalMomentMatch, InverseGammaMomentMatch, LogNormalMomentMatch
export FixedDistributionProposal, MixedProposal

# Negative Binomial model variants
export NBGammaPoissonGlobalRMarg, NBGammaPoissonGlobalRMargState, NBGammaPoissonGlobalRMargPriors, NBGammaPoissonGlobalRMargSamples
export NBGammaPoissonGlobalR, NBGammaPoissonGlobalRState, NBGammaPoissonGlobalRPriors, NBGammaPoissonGlobalRSamples
export NBGammaPoissonClusterRMarg, NBGammaPoissonClusterRMargState, NBGammaPoissonClusterRMargPriors, NBGammaPoissonClusterRMargSamples
export NBMeanDispersionGlobalR, NBMeanDispersionGlobalRState, NBMeanDispersionGlobalRPriors, NBMeanDispersionGlobalRSamples
export NBMeanDispersionClusterR, NBMeanDispersionClusterRState, NBMeanDispersionClusterRPriors, NBMeanDispersionClusterRSamples

# Poisson model variants
export PoissonClusterRates, PoissonClusterRatesState, PoissonClusterRatesPriors, PoissonClusterRatesSamples
export PoissonClusterRatesMarg, PoissonClusterRatesMargState, PoissonClusterRatesMargPriors, PoissonClusterRatesMargSamples
export PoissonPopulationRates, PoissonPopulationRatesState, PoissonPopulationRatesPriors, PoissonPopulationRatesSamples

# Binomial model variants
export BinomialClusterProb, BinomialClusterProbState, BinomialClusterProbPriors, BinomialClusterProbSamples
export BinomialClusterProbMarg, BinomialClusterProbMargState, BinomialClusterProbMargPriors, BinomialClusterProbMargSamples

# Skew Normal model variants
export SkewNormalModel, SkewNormalCluster
export SkewNormalClusterState, SkewNormalClusterPriors, SkewNormalClusterSamples

# Gamma model variants
export GammaModel, GammaClusterShapeMarg
export GammaClusterShapeMargState, GammaClusterShapeMargPriors, GammaClusterShapeMargSamples
export ContinuousData
export skewnormal_logpdf, delta_from_alpha, sample_h_conditional
export estimate_skewness, alpha_from_skewness, estimate_skewnormal_params
export update_h!, update_ξ!, update_ω!, update_α!

# Weibull model variants
export WeibullModel, WeibullCluster
export WeibullClusterState, WeibullClusterPriors, WeibullClusterSamples
export weibull_logpdf, fit_weibull_shape_moments
export update_k!, update_λ!

# DDCRP parameters
export DDCRPParams

# MCMC
export MCMCOptions, MCMCSamples, mcmc
export MCMCDiagnostics, MCMCSummary
export should_infer, get_prop_sd

# Interface methods
export table_contribution, posterior
export update_λ!, update_r!, update_m!
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

# Core DDCRP utilities
export construct_distance_matrix, decay
export precompute_log_ddcrp, ddcrp_contribution
export table_vector, table_vector_minus_i
export compute_table_assignments, table_assignments_to_vector
export simulate_ddcrp
export get_cluster_labels, c_to_z

# State utilities
export m_dict_to_samples
# Note: likelihood_contribution now in models/negative_binomial/nb_utils.jl
# Note: negbin_logpdf now in models/negative_binomial/nb_utils.jl
# Note: logbinomial now in models/binomial/binomial_utils.jl
export likelihood_contribution, negbin_logpdf, logbinomial

# Simulation utilities
export simulate_m, simulate_λ
export simulate_negbin_data, simulate_poisson_data, simulate_binomial_data, simulate_skewnormal_data
export simulate_gamma_data, simulate_weibull_data

# Analysis utilities
export calculate_n_clusters, posterior_num_cluster_distribution
export compute_similarity_matrix, compute_ari_trace, compute_vi_trace, compute_kl_ppd
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
include("core/state.jl")
include("core/options.jl")
include("core/rjmcmc_interface.jl")

# Inference machinery - proposals and diagnostics (depends on core)
include("inference/proposals.jl")
include("inference/diagnostics.jl")

# Model implementations - Negative Binomial variants
# (must be loaded before samplers that have specialized methods for them)
include("models/negative_binomial/nb_utils.jl")
include("models/negative_binomial/nb_gamma_poisson_global_r_marg.jl")
include("models/negative_binomial/nb_gamma_poisson_global_r.jl")
include("models/negative_binomial/nb_gamma_poisson_cluster_r_marg.jl")
include("models/negative_binomial/nb_mean_dispersion_global_r.jl")
include("models/negative_binomial/nb_mean_dispersion_cluster_r.jl")

# Model implementations - Poisson variants
include("models/poisson/poisson_cluster_rates.jl")
include("models/poisson/poisson_cluster_rates_marg.jl")
include("models/poisson/poisson_population_rates.jl")

# Model implementations - Binomial variants
include("models/binomial/binomial_utils.jl")
include("models/binomial/binomial_cluster_prob.jl")
include("models/binomial/binomial_cluster_prob_marg.jl")

# Model implementations - Skew Normal variants
include("models/skew_normal/skew_normal_utils.jl")
include("models/skew_normal/skew_normal_cluster.jl")

# Model implementations - Gamma variants
include("models/gamma/gamma_cluster_shape_marg.jl")

# Model implementations - Weibull variants
include("models/weibull/weibull_utils.jl")
include("models/weibull/weibull_cluster.jl")

# Samplers (depends on model types for specialized methods)
include("samplers/gibbs.jl")
include("samplers/rjmcmc.jl")

# MCMC loop (depends on models and samplers)
include("inference/mcmc.jl")

# Utilities (depends on core)
include("utils/simulation.jl")
include("utils/analysis.jl")

end # module
