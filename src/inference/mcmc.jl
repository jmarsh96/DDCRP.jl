# ============================================================================
# Generic MCMC Loop
# Dispatches on model type with update_params! handling parameter updates
# and update_c! handling assignment updates
# ============================================================================

using Random, StatsBase

# ============================================================================
# update_c! - Generic assignment update dispatcher
# ============================================================================

"""
    update_c!(model, state, data, priors, proposal, log_DDCRP, opts)

Generic assignment update dispatcher. Uses Gibbs sampling when the model
is marginalised or the proposal is conjugate; otherwise uses RJMCMC.

Returns a diagnostics vector of (move_type, i, j_star, accepted) tuples.
"""
function update_c!(
    model::LikelihoodModel,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    proposal::BirthProposal,
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    diagnostics = Vector{Tuple{Symbol, Int, Int, Bool}}()

    if !should_infer(opts, :c)
        return diagnostics
    end

    use_gibbs = proposal isa ConjugateProposal

    for i in 1:nobs(data)
        if use_gibbs
            move_type, j_star, accepted = update_c_gibbs!(model, i, state, data, priors, log_DDCRP)
        else
            move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, proposal, log_DDCRP, opts)
        end
        push!(diagnostics, (move_type, i, j_star, accepted))
    end

    return diagnostics
end

# ============================================================================
# Generic MCMC Entry Point
# ============================================================================

"""
    mcmc(model, data, ddcrp_params, priors, proposal; opts) -> MCMCSamples

Main MCMC entry point. Dispatches based on model type.

# Arguments
- `model::LikelihoodModel`: The likelihood model (determines parameter structure)
- `data::AbstractObservedData`: Observed data container
- `ddcrp_params::DDCRPParams`: DDCRP hyperparameters
- `priors::AbstractPriors`: Prior specification
- `proposal::BirthProposal`: Birth proposal for RJMCMC (or ConjugateProposal for Gibbs)

# Keyword Arguments
- `opts::MCMCOptions`: MCMC configuration

# Returns
- `MCMCSamples`: Posterior samples
- `MCMCDiagnostics` (optional): If opts.track_diagnostics is true
"""
function mcmc(
    model::LikelihoodModel,
    data::AbstractObservedData,
    ddcrp_params::DDCRPParams,
    priors::AbstractPriors,
    proposal::BirthProposal = PriorProposal();
    opts::MCMCOptions = MCMCOptions()
)
    # Validate data matches model requirements
    if requires_trials(model) && !has_trials(data)
        throw(ArgumentError(
            "Model $(typeof(model)) requires data with trials (N). " *
            "Use CountDataWithTrials(y, N, D) instead of CountData(y, D)."
        ))
    end

    n = nobs(data)
    D = distance_matrix(data)

    # Precompute DDCRP matrix
    log_DDCRP = precompute_log_ddcrp(
        ddcrp_params.decay_fn,
        ddcrp_params.α,
        ddcrp_params.scale,
        D
    )

    # Initialize state (dispatches on model type)
    state = initialise_state(model, data, ddcrp_params, priors)

    # Allocate sample storage (dispatches on model type)
    samples = allocate_samples(model, opts.n_samples, n)

    # Initialize diagnostics
    diag = opts.track_diagnostics ?
           MCMCDiagnostics(n; track_pairwise=opts.track_pairwise) : nothing

    # Store initial state
    extract_samples!(model, state, samples, 1)
    samples.logpost[1] = posterior(model, data, state, priors, log_DDCRP)

    # Main MCMC loop
    for iter in 2:opts.n_samples
        tables = table_vector(state.c)

        # Update model parameters (dispatches on model type)
        update_params!(model, state, data, priors, tables, log_DDCRP, opts)

        # Update customer assignments (gibbs or rjmcmc based on model/proposal)
        result = update_c!(model, state, data, priors, proposal, log_DDCRP, opts)

        # Record diagnostics if returned
        if opts.track_diagnostics && !isnothing(diag) && !isnothing(result)
            for (move_type, i, j_star, accepted) in result
                record_move!(diag, move_type, accepted)
                if opts.track_pairwise
                    record_pairwise!(diag, i, j_star, accepted)
                end
            end
        end

        # Store samples
        extract_samples!(model, state, samples, iter)
        samples.logpost[iter] = posterior(model, data, state, priors, log_DDCRP)

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
# Convenience Wrappers (construct data objects from y, D)
# ============================================================================

"""Convenience: CountData models (Poisson, NegBin) with separate y, D."""
function mcmc(model::LikelihoodModel, y::AbstractVector, D::AbstractMatrix,
              ddcrp_params::DDCRPParams, priors::AbstractPriors,
              proposal::BirthProposal = PriorProposal(); opts::MCMCOptions = MCMCOptions())
    data = CountData(y, D)
    return mcmc(model, data, ddcrp_params, priors, proposal; opts=opts)
end

"""Convenience: CountDataWithTrials models (Binomial, PoissonPopulationRates) with separate y, N, D."""
function mcmc(model::LikelihoodModel, y::AbstractVector, N::Union{Int, AbstractVector{Int}},
              D::AbstractMatrix, ddcrp_params::DDCRPParams, priors::AbstractPriors,
              proposal::BirthProposal = PriorProposal(); opts::MCMCOptions = MCMCOptions())
    data = CountDataWithTrials(y, N, D)
    return mcmc(model, data, ddcrp_params, priors, proposal; opts=opts)
end

"""Convenience: ContinuousData models (SkewNormal, Gamma) with separate y, D."""
function mcmc(model::Union{SkewNormalModel, GammaModel}, y::AbstractVector{<:Real},
              D::AbstractMatrix, ddcrp_params::DDCRPParams, priors::AbstractPriors,
              proposal::BirthProposal = PriorProposal(); opts::MCMCOptions = MCMCOptions())
    data = ContinuousData(y, D)
    return mcmc(model, data, ddcrp_params, priors, proposal; opts=opts)
end
