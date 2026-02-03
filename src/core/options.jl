# ============================================================================
# MCMCOptions - Configuration for MCMC sampling
# ============================================================================

"""
    MCMCOptions

Configuration for MCMC sampling.

# Fields
- `n_samples::Int`: Number of MCMC iterations (default: 10000)
- `verbose::Bool`: Print progress (default: false)
- `infer_params::Dict{Symbol, Bool}`: Which parameters to infer (default: all true)
- `prop_sds::Dict{Symbol, Float64}`: Proposal standard deviations per parameter
- `assignment_method::Symbol`: How to update assignments (:auto, :gibbs, :rjmcmc)
- `birth_proposal::Symbol`: Birth proposal type for RJMCMC (:prior, :normal_mean, :moment_matched, :lognormal)
- `birth_proposal_params::Dict{Symbol, Any}`: Parameters for birth proposal
- `fixed_dim_mode::Symbol`: Fixed-dimension mode for RJMCMC (:none, :weighted_mean, :resample_posterior)
- `track_diagnostics::Bool`: Track acceptance rates (default: true)
- `track_pairwise::Bool`: Track pairwise proposals (default: false)
"""
struct MCMCOptions
    n_samples::Int
    verbose::Bool
    infer_params::Dict{Symbol, Bool}
    prop_sds::Dict{Symbol, Float64}
    assignment_method::Symbol
    birth_proposal::Union{Symbol, UnivariateDistribution}
    birth_proposal_params::Dict{Symbol, Any}
    fixed_dim_mode::Symbol
    track_diagnostics::Bool
    track_pairwise::Bool
end

# Default constructor
function MCMCOptions(;
    n_samples::Int = 10000,
    verbose::Bool = false,
    infer_params::Dict{Symbol, Bool} = Dict{Symbol, Bool}(
        :λ => true,
        :r => true,
        :m => true,
        :p => true,
        :c => true
    ),
    prop_sds::Dict{Symbol, Float64} = Dict{Symbol, Float64}(
        :λ => 0.5,
        :r => 0.5,
        :m => 0.5,
        :p => 0.5
    ),
    assignment_method::Symbol = :auto,
    birth_proposal::Union{Symbol, UnivariateDistribution} = :prior,
    birth_proposal_params::Dict{Symbol, Any} = Dict{Symbol, Any}(),
    fixed_dim_mode::Symbol = :none,
    track_diagnostics::Bool = true,
    track_pairwise::Bool = false
)
    return MCMCOptions(
        n_samples,
        verbose,
        infer_params,
        prop_sds,
        assignment_method,
        birth_proposal,
        birth_proposal_params,
        fixed_dim_mode,
        track_diagnostics,
        track_pairwise
    )
end

"""
    should_infer(opts::MCMCOptions, param::Symbol) -> Bool

Check if a parameter should be inferred. Defaults to true if not specified.
"""
should_infer(opts::MCMCOptions, param::Symbol) = get(opts.infer_params, param, true)

"""
    get_prop_sd(opts::MCMCOptions, param::Symbol; default=0.5) -> Float64

Get the proposal standard deviation for a parameter.
"""
get_prop_sd(opts::MCMCOptions, param::Symbol; default=0.5) = get(opts.prop_sds, param, default)

"""
    build_birth_proposal(opts::MCMCOptions) -> BirthProposal

Construct a BirthProposal from MCMCOptions settings.
"""
function build_birth_proposal(opts::MCMCOptions)
    bp_type = opts.birth_proposal
    params = opts.birth_proposal_params

    if bp_type == :prior
        return PriorProposal()
    elseif bp_type == :normal_mean
        σ_mode = get(params, :σ_mode, :empirical)
        σ_fixed = get(params, :σ_fixed, 1.0)
        return NormalMeanProposal(σ_mode, σ_fixed)
    elseif bp_type == :moment_matched
        min_size = get(params, :min_size, 3)
        return MomentMatchedProposal(min_size)
    elseif bp_type == :lognormal
        σ_mode = get(params, :σ_mode, :empirical)
        σ_fixed = get(params, :σ_fixed, 1.0)
        return LogNormalProposal(σ_mode, σ_fixed)
    elseif bp_type isa UnivariateDistribution
        return bp_type
    else
        error("Unknown birth proposal type: $bp_type")
    end
end

"""
    determine_assignment_method(model::LikelihoodModel, opts::MCMCOptions) -> Symbol

Determine the assignment update method to use. If opts.assignment_method is :auto,
uses :gibbs for marginalised models and :rjmcmc for unmarginalised models.
"""
function determine_assignment_method(model::LikelihoodModel, opts::MCMCOptions)
    if opts.assignment_method == :auto
        return is_marginalised(model) ? :gibbs : :rjmcmc
    else
        # Validate that Gibbs is only used with marginalised models
        if opts.assignment_method == :gibbs && !is_marginalised(model)
            error("Gibbs assignment method requires a marginalised model, but $(typeof(model)) is not marginalised")
        end
        return opts.assignment_method
    end
end
