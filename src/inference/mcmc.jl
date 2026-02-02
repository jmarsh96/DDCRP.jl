# ============================================================================
# Generic MCMC Loop - New Architecture
# Dispatches on model type with optional AssignmentProposal
# ============================================================================

using Random, StatsBase

"""
    MCMCOptions

Configuration for MCMC sampling.

# Fields
- `n_samples::Int`: Number of MCMC iterations (default: 10000)
- `verbose::Bool`: Print progress (default: false)
- `infer_λ::Bool`: Update latent rates (default: true)
- `infer_r::Bool`: Update dispersion parameter (default: true)
- `infer_m::Bool`: Update cluster means (default: true)
- `infer_customer_assignments::Bool`: Update customer assignments (default: true)
- `track_diagnostics::Bool`: Track acceptance rates (default: true)
- `track_pairwise::Bool`: Track pairwise proposals (default: false)
- `prop_sd_λ::Float64`: Proposal std for λ updates (default: 0.5)
- `prop_sd_r::Float64`: Proposal std for r updates (default: 0.5)
- `prop_sd_m::Float64`: Proposal std for m updates (default: 0.5)
"""
Base.@kwdef struct MCMCOptions
    n_samples::Int = 10000
    verbose::Bool = false

    # Inference toggles
    infer_λ::Bool = true
    infer_r::Bool = true
    infer_m::Bool = true
    infer_customer_assignments::Bool = true

    # Diagnostics
    track_diagnostics::Bool = true
    track_pairwise::Bool = false

    # Proposal standard deviations
    prop_sd_λ::Float64 = 0.5
    prop_sd_r::Float64 = 0.5
    prop_sd_m::Float64 = 0.5
end

# Convert to NamedTuple for backwards compatibility
function Base.getindex(opts::MCMCOptions, key::Symbol)
    return getfield(opts, key)
end

function Base.get(opts::MCMCOptions, key::Symbol, default)
    if hasfield(MCMCOptions, key)
        return getfield(opts, key)
    else
        return default
    end
end

# ============================================================================
# Generic MCMC Entry Point - New Signature
# ============================================================================

"""
    mcmc(model, y, D, ddcrp_params, priors; proposal, opts) -> MCMCSamples

Main MCMC entry point. Dispatches based on model type.

# Arguments
- `model::LikelihoodModel`: The likelihood model (determines parameter structure)
- `y::Vector`: Observed data
- `D::Matrix`: Distance matrix
- `ddcrp_params::DDCRPParams`: DDCRP hyperparameters
- `priors::AbstractPriors`: Prior specification

# Keyword Arguments
- `proposal::AssignmentProposal`: How to update customer assignments (default: model-specific)
- `opts::MCMCOptions`: MCMC configuration

# Returns
- `MCMCSamples`: Posterior samples
- `MCMCDiagnostics` (optional): If opts.track_diagnostics is true
"""
function mcmc(
    model::LikelihoodModel,
    y::AbstractVector,
    D::AbstractMatrix,
    ddcrp_params::DDCRPParams,
    priors::AbstractPriors;
    proposal::AssignmentProposal = default_proposal(model),
    opts::MCMCOptions = MCMCOptions()
)
    # Validate proposal is compatible with model
    validate_proposal(model, proposal)

    n = length(y)

    # Precompute DDCRP matrix
    log_DDCRP = precompute_log_ddcrp(
        ddcrp_params.decay_fn,
        ddcrp_params.α,
        ddcrp_params.scale,
        D
    )

    # Initialize state (dispatches on model type)
    state = initialise_state(model, y, D, ddcrp_params, priors)

    # Allocate sample storage (dispatches on model type)
    samples = allocate_samples(model, opts.n_samples, n)

    # Initialize diagnostics
    diag = opts.track_diagnostics ?
           MCMCDiagnostics(n; track_pairwise=opts.track_pairwise) : nothing

    # Store initial state
    extract_samples!(model, state, samples, 1)
    samples.logpost[1] = posterior(model, y, state, priors, log_DDCRP)

    # Main MCMC loop
    for iter in 2:opts.n_samples
        tables = table_vector(state.c)

        # Update model-specific parameters (dispatches on model type)
        update_params!(model, state, y, priors, tables;
                      prop_sd_λ=opts.prop_sd_λ,
                      prop_sd_m=opts.prop_sd_m,
                      prop_sd_r=opts.prop_sd_r,
                      infer_λ=opts.infer_λ,
                      infer_m=opts.infer_m,
                      infer_r=opts.infer_r)

        # Update customer assignments (dispatches on proposal and model)
        if opts.infer_customer_assignments
            for i in eachindex(y)
                result = update_c!(proposal, model, i, state, y, priors, log_DDCRP)

                if opts.track_diagnostics && !isnothing(diag)
                    move_type, j_star, accepted = result
                    record_move!(diag, move_type, accepted)
                    if opts.track_pairwise
                        record_pairwise!(diag, i, j_star, accepted)
                    end
                end
            end
        end

        # Store samples
        extract_samples!(model, state, samples, iter)
        samples.logpost[iter] = posterior(model, y, state, priors, log_DDCRP)

        # Progress
        if opts.verbose && (iter % 100 == 0 || iter == 1)
            println("Iteration $iter / $(opts.n_samples)")
        end
    end

    if opts.track_diagnostics && !isnothing(diag)
        finalize!(diag)
        return samples, diag
    end

    return samples
end

# ============================================================================
# Convenience method accepting NamedTuple opts
# ============================================================================

function mcmc(
    model::LikelihoodModel,
    y::AbstractVector,
    D::AbstractMatrix,
    ddcrp_params::DDCRPParams,
    priors::AbstractPriors,
    opts::NamedTuple
)
    mcmc_opts = MCMCOptions(;
        n_samples = get(opts, :n_samples, 10000),
        verbose = get(opts, :verbose, false),
        infer_λ = get(opts, :infer_lambda, get(opts, :infer_λ, true)),
        infer_r = get(opts, :infer_r, true),
        infer_m = get(opts, :infer_m, true),
        infer_customer_assignments = get(opts, :infer_customer_assignments, true),
        track_diagnostics = get(opts, :track_diagnostics, true),
        track_pairwise = get(opts, :track_pairwise, false),
    )

    # Check for proposal in opts
    proposal = if haskey(opts, :proposal)
        opts.proposal
    elseif haskey(opts, :birth_proposal)
        # Legacy: convert birth_proposal to RJMCMCProposal
        RJMCMCProposal(opts.birth_proposal, get(opts, :fixed_dim_mode, :none))
    else
        default_proposal(model)
    end

    return mcmc(model, y, D, ddcrp_params, priors; proposal=proposal, opts=mcmc_opts)
end

# ============================================================================
# Backward Compatibility - Old Signature with Strategy
# ============================================================================

# Legacy InferenceStrategy types are defined in core/types.jl
# (InferenceStrategy, MarginalisedStrategy, UnmarginalisedStrategy,
#  Marginalised, Unmarginalised, RJMCMC_Strategy)

# Keep old signature working during transition
function mcmc(
    model::LikelihoodModel,
    strategy::InferenceStrategy,
    y::AbstractVector,
    D::AbstractMatrix,
    ddcrp_params::DDCRPParams,
    priors::AbstractPriors,
    opts::MCMCOptions = MCMCOptions()
)
    @warn "mcmc(model, strategy, ...) is deprecated. Use mcmc(model, y, D, ...; proposal=...) instead." maxlog=1

    # Convert old strategy to new proposal
    proposal = strategy_to_proposal(strategy, opts)

    return mcmc(model, y, D, ddcrp_params, priors; proposal=proposal, opts=opts)
end

"""
    strategy_to_proposal(strategy, opts) -> AssignmentProposal

Convert legacy strategy to new proposal type.
"""
strategy_to_proposal(::MarginalisedStrategy, opts) = GibbsProposal()
strategy_to_proposal(::Unmarginalised, opts) = MetropolisProposal()
function strategy_to_proposal(::RJMCMC_Strategy, opts)
    bp = get(opts, :birth_proposal, PriorProposal())
    fdm = get(opts, :fixed_dim_mode, :none)
    return RJMCMCProposal(bp, fdm)
end

# Legacy trait functions for backward compatibility
has_cluster_means(::MarginalisedStrategy) = false
has_cluster_means(::UnmarginalisedStrategy) = true
