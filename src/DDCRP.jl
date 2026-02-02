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

# Assignment proposal types (new)
export AssignmentProposal, GibbsProposal, MetropolisProposal, RJMCMCProposal

# Birth proposals for RJMCMC
export PriorProposal, NormalMeanProposal, MomentMatchedProposal, LogNormalProposal

# Negative Binomial model variants
export NBGammaPoissonGlobalRMarg, NBGammaPoissonGlobalRMargState, NBGammaPoissonGlobalRMargPriors
export NBGammaPoissonGlobalR, NBGammaPoissonGlobalRState, NBGammaPoissonGlobalRPriors
export NBGammaPoissonClusterRMarg, NBGammaPoissonClusterRMargState, NBGammaPoissonClusterRMargPriors
export NBMeanDispersionGlobalR, NBMeanDispersionGlobalRState, NBMeanDispersionGlobalRPriors
export NBMeanDispersionClusterR, NBMeanDispersionClusterRState, NBMeanDispersionClusterRPriors

# Poisson model variants
export PoissonClusterRates, PoissonClusterRatesState, PoissonClusterRatesPriors
export PoissonClusterRatesMarg, PoissonClusterRatesMargState, PoissonClusterRatesMargPriors
export PoissonPopulationRates, PoissonPopulationRatesState, PoissonPopulationRatesPriors

# Binomial model variants
export BinomialClusterProb, BinomialClusterProbState, BinomialClusterProbPriors
export BinomialClusterProbMarg, BinomialClusterProbMargState, BinomialClusterProbMargPriors

# DDCRP parameters
export DDCRPParams

# MCMC
export MCMCOptions, MCMCSamples, mcmc
export MCMCDiagnostics, MCMCSummary

# Interface methods
export table_contribution, posterior
export update_λ!, update_r!, update_m!, update_c!
export update_params!, update_cluster_rates!, update_cluster_probs!
export sample_proposal, proposal_logpdf
export initialise_state, extract_samples!, allocate_samples

# Trait functions
export has_latent_rates, has_global_dispersion, has_cluster_dispersion
export has_cluster_means, has_cluster_rates, has_cluster_probs
export is_marginalised, default_proposal, validate_proposal

# Diagnostics
export acceptance_rates, pairwise_acceptance_rates
export effective_sample_size, integrated_autocorrelation_time, ess_per_second
export autocorrelation, summarize_mcmc
export record_move!, record_pairwise!, finalize!

# Core DDCRP utilities
export construct_distance_matrix, decay
export precompute_log_ddcrp, ddcrp_contribution
export table_vector, table_vector_minus_i
export compute_table_assignments, table_assignments_to_vector
export simulate_ddcrp
export get_cluster_labels, c_to_z

# State utilities
export m_dict_to_samples, likelihood_contribution

# Simulation utilities
export simulate_m, simulate_λ
export simulate_negbin_data, simulate_poisson_data, simulate_binomial_data

# Analysis utilities
export calculate_n_clusters, posterior_num_cluster_distribution
export compute_similarity_matrix, compute_ari_trace
export point_estimate_clustering, posterior_summary

# Sampler utilities
export get_moving_set, find_table_for_customer
export compute_fixed_dim_means, compute_weighted_means, resample_posterior_means

# Proposal utilities
export fit_inverse_gamma_moments, compute_proposal_σ, compute_lognormal_σ

# Legacy exports (backward compatibility)
export InferenceStrategy, MarginalisedStrategy, UnmarginalisedStrategy
export Marginalised, Unmarginalised, RJMCMC_Strategy

# ============================================================================
# Include files in dependency order
# ============================================================================

# Core types and infrastructure (no dependencies)
include("core/types.jl")
include("core/priors.jl")
include("core/ddcrp.jl")
include("core/state.jl")

# Inference machinery - proposals and diagnostics (depends on core)
include("inference/proposals.jl")
include("inference/diagnostics.jl")

# Model implementations - Negative Binomial variants
# (must be loaded before samplers that have specialized methods for them)
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
include("models/binomial/binomial_cluster_prob.jl")
include("models/binomial/binomial_cluster_prob_marg.jl")

# Samplers (depends on model types for specialized methods)
include("samplers/gibbs.jl")
include("samplers/rjmcmc.jl")

# MCMC loop (depends on models and samplers)
include("inference/mcmc.jl")

# Utilities (depends on core)
include("utils/simulation.jl")
include("utils/analysis.jl")

end # module
