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
    update_c!(model, state, data, priors, birth_proposal, fixed_dim_proposal, log_DDCRP, opts)

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
    fixed_dim_proposal::FixedDimensionProposal,
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions;
    missing_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)
    diagnostics = Vector{Tuple{Symbol, Int, Int, Bool}}()

    if !should_infer(opts, :c)
        return diagnostics
    end

    use_gibbs = proposal isa ConjugateProposal
    y_data = observations(data)

    for i in 1:nobs(data)
        if !isnothing(missing_mask) && missing_mask[i]
            # Missing observation: sample assignment from DDCRP prior only,
            # then impute y[i] from the cluster's posterior predictive.
            y_data[i] = missing
            if use_gibbs
                move_type, j_star, accepted = update_c_gibbs!(model, i, state, data, priors, log_DDCRP)
            else
                move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, proposal, fixed_dim_proposal, log_DDCRP)
            end
            y_data[i] = impute_y(model, i, state, data, priors)
        else
            if use_gibbs
                move_type, j_star, accepted = update_c_gibbs!(model, i, state, data, priors, log_DDCRP)
            else
                move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, proposal, fixed_dim_proposal, log_DDCRP)
            end
        end
        push!(diagnostics, (move_type, i, j_star, accepted))
    end

    return diagnostics
end

# ============================================================================
# Generic MCMC Entry Point
# ============================================================================

"""
    mcmc(model, data, ddcrp_params, priors, proposal; fixed_dim_proposal, opts) -> MCMCSamples

Main MCMC entry point. Dispatches based on model type.

# Arguments
- `model::LikelihoodModel`: The likelihood model (determines parameter structure)
- `data::AbstractObservedData`: Observed data container
- `ddcrp_params::DDCRPParams`: DDCRP hyperparameters
- `priors::AbstractPriors`: Prior specification
- `proposal::BirthProposal`: Birth proposal for RJMCMC (or ConjugateProposal for Gibbs)

# Keyword Arguments
- `fixed_dim_proposal::FixedDimensionProposal`: Fixed-dimension proposal (default: `NoUpdate()`)
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
    fixed_dim_proposal::FixedDimensionProposal = NoUpdate(),
    opts::MCMCOptions = MCMCOptions(),
    init_params::Union{Nothing, Dict{Symbol,Any}} = nothing
)
    # Validate data matches model requirements
    if requires_trials(model) && !has_trials(data)
        throw(ArgumentError(
            "Model $(typeof(model)) requires data with trials (N). " *
            "Use CountDataWithTrials(y, N, D) instead of CountData(y, D)."
        ))
    end
    if requires_population(model) && !has_population(data)
        throw(ArgumentError(
            "Model $(typeof(model)) requires population data (P). " *
            "Use CountDataWithPopulation(y, P, D) instead of CountData(y, D)."
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
    if !isnothing(init_params)
        for (k, v) in init_params
            setfield!(state, k, v isa AbstractArray ? copy(v) : v)
        end
    end

    # Detect missing observations
    y_raw = observations(data)
    has_missing = any(ismissing, y_raw)
    missing_indices = has_missing ? findall(ismissing, y_raw) : Int[]
    missing_mask = has_missing ? ismissing.(y_raw) : nothing

    # Allocate sample storage (dispatches on model type)
    samples = allocate_samples(model, opts.n_samples, n, missing_indices)

    # Initial imputation: give update_params! valid y values on first iteration
    if has_missing
        for i in missing_indices
            y_raw[i] = impute_y(model, i, state, data, priors)
        end
    end

    # Initialize diagnostics
    diag = opts.track_diagnostics ?
           MCMCDiagnostics(n; track_pairwise=opts.track_pairwise) : nothing

    # Store initial state
    extract_samples!(model, state, samples, 1)
    samples.logpost[1] = posterior(model, data, state, priors, log_DDCRP)

    # Store initial imputed values
    if has_missing
        for (k, i) in enumerate(missing_indices)
            samples.y_imp[1, k] = Float64(y_raw[i])
        end
    end

    # DDCRP hyperparameter inference setup
    α_current = ddcrp_params.α
    s_current = ddcrp_params.scale
    infer_α = should_infer(opts, :α_ddcrp) && !isnothing(ddcrp_params.α_a)
    infer_s = should_infer(opts, :s_ddcrp) && !isnothing(ddcrp_params.s_a)
    infer_ddcrp = infer_α || infer_s
    V = infer_ddcrp ? zeros(n) : Float64[]
    samples.α_ddcrp[1] = α_current
    samples.s_ddcrp[1] = s_current

    # Main MCMC loop
    for iter in 2:opts.n_samples
        tables = table_vector(state.c)

        # Update model parameters (dispatches on model type)
        update_params!(model, state, data, priors, tables, log_DDCRP, opts)

        # Update customer assignments (gibbs or rjmcmc based on model/proposal)
        result = update_c!(model, state, data, priors, proposal, fixed_dim_proposal, log_DDCRP, opts;
                           missing_mask=missing_mask)

        # Update DDCRP hyperparameters if requested
        if infer_ddcrp
            R = compute_R(s_current, D)
            sample_V!(V, α_current, R)
            if infer_α
                n_self = count_self_links(state.c)
                α_current = update_α_ddcrp(n_self, V, ddcrp_params)
            end
            if infer_s
                prop_sd_s = get_prop_sd(opts, :s_ddcrp)
                if infer_α
                    s_current = update_s_ddcrp_augmented(s_current, V, R, state.c, D, ddcrp_params, prop_sd_s)
                else
                    s_current = update_s_ddcrp(s_current, α_current, state.c, D, ddcrp_params, prop_sd_s)
                end
            end
            log_DDCRP = precompute_log_ddcrp(ddcrp_params.decay_fn, α_current, s_current, D)
        end

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
        samples.α_ddcrp[iter] = α_current
        samples.s_ddcrp[iter] = s_current

        # Record imputed values for missing observations
        if has_missing
            for (k, i) in enumerate(missing_indices)
                samples.y_imp[iter, k] = Float64(y_raw[i])
            end
        end

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
              proposal::BirthProposal = PriorProposal();
              fixed_dim_proposal::FixedDimensionProposal = NoUpdate(),
              opts::MCMCOptions = MCMCOptions())
    data = CountData(y, D)
    return mcmc(model, data, ddcrp_params, priors, proposal;
                fixed_dim_proposal=fixed_dim_proposal, opts=opts)
end

"""Convenience: CountDataWithTrials models (Binomial) or CountDataWithPopulation models with separate y, N/P, D."""
function mcmc(model::LikelihoodModel, y::AbstractVector, N::Union{<:Real, <:AbstractVector{<:Real}},
              D::AbstractMatrix, ddcrp_params::DDCRPParams, priors::AbstractPriors,
              proposal::BirthProposal = PriorProposal();
              fixed_dim_proposal::FixedDimensionProposal = NoUpdate(),
              opts::MCMCOptions = MCMCOptions(),
              init_params::Union{Nothing, Dict{Symbol,Any}} = nothing)
    if requires_population(model)
        data = CountDataWithPopulation(y, N, D)
    else
        data = CountDataWithTrials(y, N, D)
    end
    return mcmc(model, data, ddcrp_params, priors, proposal;
                fixed_dim_proposal=fixed_dim_proposal, opts=opts, init_params=init_params)
end

"""Convenience: ContinuousData models (SkewNormal, Gamma, Weibull) with separate y, D."""
function mcmc(model::Union{SkewNormalModel, GammaModel, WeibullModel}, y::AbstractVector{<:Real},
              D::AbstractMatrix, ddcrp_params::DDCRPParams, priors::AbstractPriors,
              proposal::BirthProposal = PriorProposal();
              fixed_dim_proposal::FixedDimensionProposal = NoUpdate(),
              opts::MCMCOptions = MCMCOptions())
    data = ContinuousData(y, D)
    return mcmc(model, data, ddcrp_params, priors, proposal;
                fixed_dim_proposal=fixed_dim_proposal, opts=opts)
end
