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
    opts::MCMCOptions,
    missing_mask::BitVector
)
    diagnostics = Vector{Tuple{Symbol, Int, Int, Bool}}()

    if !should_infer(opts, :c)
        return diagnostics
    end

    
    for i in 1:nobs(data)
        update_type = missing_mask[i] ? MissingUpdate() : StandardUpdate()
        move_type, j_star, accepted = update_c_i!(model, i, state, data, priors, log_DDCRP, proposal, fixed_dim_proposal, update_type)
        push!(diagnostics, (move_type, i, j_star, accepted))
    end

    return diagnostics
end

# Gibbs implementation to update c
function update_c_i!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix,
    ::ConjugateProposal,
    ::FixedDimensionProposal,
    ::StandardUpdate
)
    n = length(state.c)
    log_probs = zeros(n)

    table_minus_i = table_vector_minus_i(i, state.c)
    current_table = table_minus_i[findfirst(x -> i in x, table_minus_i)]

    current_table_contrib = table_contribution(model, current_table, state, data, priors)

    for customer in current_table
        log_probs[customer] = log_DDCRP[i, customer]
    end

    remaining_tables = setdiff(table_minus_i, [current_table])

    for table in remaining_tables
        prop_table_contrib = table_contribution(model, table, state, data, priors)
        joined_table = vcat(current_table, table)
        joined_table_contrib = table_contribution(model, joined_table, state, data, priors)

        term = joined_table_contrib - prop_table_contrib - current_table_contrib

        for customer in table
            log_probs[customer] = log_DDCRP[i, customer] + term
        end
    end

    probs = exp.(log_probs .- maximum(log_probs))
    new_assignment = sample(1:n, Weights(probs))
    state.c[i] = new_assignment

    return (:gibbs, new_assignment, true)
end

function update_c_i!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix,
    birth_proposal::ConjugateProposal,
    fixed_dim_proposal::FixedDimensionProposal,
    ::MissingUpdate
)
    n = length(state.c)
    log_probs = zeros(n)
    for customer in eachindex(state.c)
        log_probs[customer] = log_DDCRP[i, customer]
    end
    probs = exp.(log_probs .- maximum(log_probs))
    new_assignment = sample(1:n, Weights(probs))
    state.c[i] = new_assignment
    data.y[i] = impute_y(model, state, data, priors, i)

    return (:gibbs, new_assignment, true)
end

# standard RJMCMC update
function update_c_i!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix,
    proposal::BirthProposal,
    fixed_dim_proposal::FixedDimensionProposal,
    ::StandardUpdate
)
    n = length(state.c)
    j_old = state.c[i]

    dicts = cluster_param_dicts(state)
    primary_dict = first(dicts)
    table_Si = find_table_for_customer(i, primary_dict)
    S_i = get_moving_set(i, state.c, table_Si)
    # table_l deferred to birth branch — avoids setdiff on every call

    j_star = rand(1:n)

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    # Delta DDCRP: only c[i] changes, so delta is O(1)
    ddcrp_delta = log_DDCRP[i, j_star] - log_DDCRP[i, j_old]

    if !j_old_in_Si && j_star_in_Si
        # ===== BIRTH MOVE =====
        # Sample birth params before modifying state
        params_new, log_q_forward = sample_birth_params(model, proposal, S_i, state, data, priors)

        # S_i already sorted (from table_assignments_to_vector)
        # table_Si already sorted (dict key)
        table_l = sorted_setdiff(table_Si, S_i)

        # Compute old table contribution before modification
        old_contrib = table_contribution(model, table_Si, state, data, priors)

        # Save affected entries
        saved = save_entries(dicts, [table_Si])
        old_table_Si_vals = map(d -> d[table_Si], dicts)

        # Modify state in-place
        state.c[i] = j_star
        for k in keys(dicts)
            dicts[k][S_i] = params_new[k]
            if !isempty(table_l)
                dicts[k][table_l] = old_table_Si_vals[k]
            end
            delete!(dicts[k], table_Si)
        end

        # Compute new table contributions
        new_contrib = table_contribution(model, S_i, state, data, priors)
        if !isempty(table_l)
            new_contrib += table_contribution(model, table_l, state, data, priors)
        end

        lpr = -log_q_forward
        log_α = (new_contrib - old_contrib) + ddcrp_delta + lpr

        if log(rand()) < log_α
            return (:birth, j_star, true)
        else
            # Restore state
            state.c[i] = j_old
            keys_to_delete = isempty(table_l) ? [S_i] : [S_i, table_l]
            restore_entries!(dicts, saved, keys_to_delete)
            return (:birth, j_star, false)
        end

    elseif j_old_in_Si && !j_star_in_Si
        # ===== DEATH MOVE =====
        table_target = find_table_for_customer(j_star, primary_dict)
        merged_table = sorted_merge(table_Si, table_target)

        # Reverse proposal density (before modification)
        params_old = map(d -> d[table_Si], dicts)
        log_q_reverse = birth_params_logpdf(model, proposal, params_old, S_i, state, data, priors)
        lpr = log_q_reverse

        # Compute old contributions before modification
        old_contrib = table_contribution(model, table_Si, state, data, priors) +
                      table_contribution(model, table_target, state, data, priors)

        # Save affected entries
        saved = save_entries(dicts, [table_Si, table_target])
        target_vals = map(d -> d[table_target], dicts)

        # Modify state in-place
        state.c[i] = j_star
        for k in keys(dicts)
            dicts[k][merged_table] = target_vals[k]
            delete!(dicts[k], table_Si)
            delete!(dicts[k], table_target)
        end

        # Compute new contribution
        new_contrib = table_contribution(model, merged_table, state, data, priors)

        log_α = (new_contrib - old_contrib) + ddcrp_delta + lpr

        if log(rand()) < log_α
            return (:death, j_star, true)
        else
            state.c[i] = j_old
            restore_entries!(dicts, saved, [merged_table])
            return (:death, j_star, false)
        end

    else
        # ===== FIXED DIMENSION MOVE =====
        table_old_target = find_table_for_customer(j_old, primary_dict)
        table_new_target = find_table_for_customer(j_star, primary_dict)

        if table_old_target == table_new_target
            # Same table - only c[i] changes, no table contribution changes
            log_α = ddcrp_delta

            if log(rand()) < log_α
                state.c[i] = j_star
                return (:fixed, j_star, true)
            end
            return (:fixed, j_star, false)
        end

        # Different tables - transfer S_i from old to new table
        new_table_depleted = sorted_setdiff(table_old_target, S_i)
        new_table_augmented = sorted_merge(table_new_target, S_i)

        # Compute fixed-dim params before modification
        params_depl, params_aug, lpr = fixed_dim_params(
            model, fixed_dim_proposal, S_i, table_old_target, table_new_target, state, data, priors)

        # Compute old contributions before modification
        old_contrib = table_contribution(model, table_old_target, state, data, priors) +
                      table_contribution(model, table_new_target, state, data, priors)

        # Save affected entries
        saved = save_entries(dicts, [table_old_target, table_new_target])

        # Modify state in-place
        state.c[i] = j_star
        for k in keys(dicts)
            if !isempty(new_table_depleted)
                dicts[k][new_table_depleted] = params_depl[k]
            end
            dicts[k][new_table_augmented] = params_aug[k]
            delete!(dicts[k], table_old_target)
            delete!(dicts[k], table_new_target)
        end

        # Compute new contributions
        new_contrib = table_contribution(model, new_table_augmented, state, data, priors)
        if !isempty(new_table_depleted)
            new_contrib += table_contribution(model, new_table_depleted, state, data, priors)
        end

        log_α = (new_contrib - old_contrib) + ddcrp_delta + lpr

        if log(rand()) < log_α
            return (:fixed, j_star, true)
        else
            state.c[i] = j_old
            keys_to_delete = isempty(new_table_depleted) ?
                [new_table_augmented] : [new_table_depleted, new_table_augmented]
            restore_entries!(dicts, saved, keys_to_delete)
            return (:fixed, j_star, false)
        end
    end
end


# RJMCMC update for missing observations.
function update_c_i!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix,
    proposal::BirthProposal,
    ::FixedDimensionProposal,
    ::MissingUpdate
)
    n = length(state.c)
    j_old = state.c[i]

    dicts = cluster_param_dicts(state)
    primary_dict = first(dicts)
    table_Si = find_table_for_customer(i, primary_dict)
    S_i = get_moving_set(i, state.c, table_Si)

    # Draw from DDCRP prior — no data term for missing y_i
    log_probs = [log_DDCRP[i, j] for j in 1:n]
    probs = exp.(log_probs .- maximum(log_probs))
    j_star = sample(1:n, Weights(probs))

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    if !j_old_in_Si && j_star_in_Si
        # ===== BIRTH MOVE — always accept =====
        # Sample new cluster params from birth proposal (e.g. conjugate posterior
        # for NBPopulationRates); accept unconditionally.
        params_new, _ = sample_birth_params(model, proposal, S_i, state, data, priors)

        table_l = sorted_setdiff(table_Si, S_i)
        old_table_Si_vals = map(d -> d[table_Si], dicts)

        state.c[i] = j_star
        for k in keys(dicts)
            dicts[k][S_i] = params_new[k]
            if !isempty(table_l)
                dicts[k][table_l] = old_table_Si_vals[k]
            end
            delete!(dicts[k], table_Si)
        end

    elseif j_old_in_Si && !j_star_in_Si
        # ===== DEATH MOVE — always accept =====
        # Merge S_i into target cluster, inherit target cluster params.
        table_target = find_table_for_customer(j_star, primary_dict)
        merged_table = sorted_merge(table_Si, table_target)
        target_vals = map(d -> d[table_target], dicts)

        state.c[i] = j_star
        for k in keys(dicts)
            dicts[k][merged_table] = target_vals[k]
            delete!(dicts[k], table_Si)
            delete!(dicts[k], table_target)
        end

    else
        # ===== FIXED DIMENSION MOVE — always accept =====
        table_old_target = find_table_for_customer(j_old, primary_dict)
        table_new_target = find_table_for_customer(j_star, primary_dict)

        state.c[i] = j_star

        if table_old_target != table_new_target
            # Transfer S_i: depleted cluster keeps old params, augmented inherits new.
            new_table_depleted  = sorted_setdiff(table_old_target, S_i)
            new_table_augmented = sorted_merge(table_new_target, S_i)
            old_depleted_vals   = map(d -> d[table_old_target], dicts)
            target_vals         = map(d -> d[table_new_target], dicts)

            for k in keys(dicts)
                if !isempty(new_table_depleted)
                    dicts[k][new_table_depleted] = old_depleted_vals[k]
                end
                dicts[k][new_table_augmented] = target_vals[k]
                delete!(dicts[k], table_old_target)
                delete!(dicts[k], table_new_target)
            end
        end
    end

    data.y[i] = impute_y(model, state, data, priors, i)
    return (:gibbs, j_star, true)
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
    init_params::Union{Nothing, Dict{Symbol,Any}} = nothing,
    missing_mask::BitVector = nothing
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

    # Allocate sample storage (dispatches on model type)
    samples = allocate_samples(model, opts.n_samples, n)

    # Initialize diagnostics
    diag = opts.track_diagnostics ?
           MCMCDiagnostics(n; track_pairwise=opts.track_pairwise) : nothing

    # Store initial state
    extract_samples!(model, state, samples, 1)
    samples.logpost[1] = posterior(model, data, state, priors, log_DDCRP)

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
        result = update_c!(model, state, data, priors, proposal, fixed_dim_proposal, log_DDCRP, opts, missing_mask)

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
function mcmc(model::LikelihoodModel, y_::AbstractVector, N::Union{<:Real, <:AbstractVector{<:Real}},
              D::AbstractMatrix, ddcrp_params::DDCRPParams, priors::AbstractPriors,
              proposal::BirthProposal = PriorProposal();
              fixed_dim_proposal::FixedDimensionProposal = NoUpdate(),
              opts::MCMCOptions = MCMCOptions(),
              init_params::Union{Nothing, Dict{Symbol,Any}} = nothing
)
    n = length(y_)
    missing_mask = ismissing.(y_)
    if sum(missing_mask) > 0
        finite_obs = collect(skipmissing(y_))
        y = zeros(Int64, n)
        for i in 1:n
            if ismissing(y_[i])
                y[i] = rand(finite_obs)
            else
                y[i] = y_[i]
            end
        end
    else
        y = y_
    end
    if requires_population(model)
        data = CountDataWithPopulation(y, N, D)
    else
        data = CountDataWithTrials(y, N, D)
    end
    return mcmc(
        model, data, ddcrp_params, priors, proposal;
        fixed_dim_proposal=fixed_dim_proposal, 
        opts=opts, 
        init_params=init_params,
        missing_mask=missing_mask
    )
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
