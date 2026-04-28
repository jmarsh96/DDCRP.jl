# ============================================================================
# Reversible Jump MCMC for Customer Assignments
# Handles trans-dimensional moves when clusters split or merge
# ============================================================================

using StatsBase, Distributions

"""
    get_moving_set(i, c)

Get the moving set S_i: the connected component containing customer i
when customer i's link is temporarily removed (set to self-loop).
"""
function get_moving_set(i, c)
    tables_minus_i = table_vector_minus_i(i, c)
    idx = findfirst(x -> i in x, tables_minus_i)
    return tables_minus_i[idx]
end

"""
    get_moving_set(i, c, table_Si)

Fast version: find S_i within the known table `table_Si` (sorted dict key).
Traces only within the table instead of recomputing all tables from scratch.
Result is sorted (preserves order from sorted `table_Si`).
"""
function get_moving_set(i::Int, c::Vector{Int}, table_Si::Vector{Int})
    n = length(c)
    # 0x00 = unknown, 0x01 = reaches i, 0x02 = doesn't, 0xff = on current path
    status = zeros(UInt8, n)
    status[i] = 0x01
    path = Int[]

    for j in table_Si
        status[j] != 0x00 && continue

        empty!(path)
        curr = j
        while status[curr] == 0x00
            push!(path, curr)
            status[curr] = 0xff
            curr = c[curr]
        end

        s = (status[curr] == 0xff) ? 0x02 : status[curr]
        for p in path
            status[p] = s
        end
    end

    return filter(j -> status[j] == 0x01, table_Si)
end

"""
    find_table_for_customer(j, param_dict)

Find the table (key in param_dict) that contains customer j.
Works with m_dict, λ_dict, p_dict, etc.
"""
function find_table_for_customer(j, param_dict)
    for (table, _) in param_dict
        if j in table
            return table
        end
    end
    error("Customer $j not found in any table")
end

# ============================================================================
# Generic RJMCMC for Customer Assignments
# Uses the interface methods from rjmcmc_interface.jl:
#   cluster_param_dicts, sample_birth_params, birth_params_logpdf, fixed_dim_params
# Uses in-place state modification with save/restore and delta-posterior
# ============================================================================

"""
    update_c_rjmcmc!(model, i, state, data, priors, birth_proposal, fixed_dim_proposal, log_DDCRP)

Generic RJMCMC update for customer i's assignment.
Dispatches on interface methods to handle any number of cluster parameter dicts.
Uses in-place state modification with save/restore instead of copying,
and delta-posterior computation instead of full posterior evaluation.

Returns (move_type::Symbol, j_star::Int, accepted::Bool)
"""
function update_c_rjmcmc!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    proposal::BirthProposal,
    fixed_dim_proposal::FixedDimensionProposal,
    log_DDCRP::AbstractMatrix
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
