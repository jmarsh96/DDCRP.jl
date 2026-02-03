# ============================================================================
# Generic MCMC Loop - New Architecture
# Dispatches on model type with update_params! handling all updates
# ============================================================================

using Random, StatsBase

# MCMCOptions is defined in core/options.jl

# ============================================================================
# update_params! - Implemented by each model
# ============================================================================
# Each model implements its own update_params!(model, state, y, priors, tables, log_DDCRP, opts)
# that handles both parameter updates and customer assignment updates based on opts.
# No fallback is provided - all models must implement this interface.

# ============================================================================
# Generic MCMC Entry Point
# ============================================================================

"""
    mcmc(model, data, ddcrp_params, priors; opts) -> MCMCSamples

Main MCMC entry point. Dispatches based on model type.

# Arguments
- `model::LikelihoodModel`: The likelihood model (determines parameter structure)
- `data::AbstractObservedData`: Observed data container (CountData or CountDataWithTrials)
- `ddcrp_params::DDCRPParams`: DDCRP hyperparameters
- `priors::AbstractPriors`: Prior specification

# Keyword Arguments
- `opts::MCMCOptions`: MCMC configuration (includes assignment_method, prop_sds, infer_params)

# Returns
- `MCMCSamples`: Posterior samples
- `MCMCDiagnostics` (optional): If opts.track_diagnostics is true
"""
function mcmc(
    model::LikelihoodModel,
    data::AbstractObservedData,
    ddcrp_params::DDCRPParams,
    priors::AbstractPriors;
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

        # Update model parameters and customer assignments (dispatches on model type)
        # Each model's update_params! handles both parameter updates and assignment updates
        result = update_params!(model, state, data, priors, tables, log_DDCRP, opts)

        # Record diagnostics if returned
        # Diagnostics format: (move_type, i, j_star, accepted)
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
