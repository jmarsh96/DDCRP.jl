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
# Fixed-Dimension Mean Update Functions
# ============================================================================

"""
    compute_fixed_dim_means(mode, S_i, λ, table_A, m_A, table_B, m_B, priors)

Compute updated means when S_i moves from cluster A to cluster B.
Returns (m_A_new, m_B_new, log_proposal_ratio).
"""
function compute_fixed_dim_means(mode::Symbol, S_i, λ, table_A, m_A, table_B, m_B, priors)
    if mode == :weighted_mean
        return compute_weighted_means(S_i, λ, table_A, m_A, table_B, m_B)
    elseif mode == :resample_posterior
        return resample_posterior_means(S_i, λ, table_A, m_A, table_B, m_B, priors)
    else
        # :none - No update, maintain existing means
        return m_A, m_B, 0.0
    end
end

"""
    compute_weighted_means(S_i, λ, table_A, m_A, table_B, m_B)

Weighted mean update: new means are weighted averages.
"""
function compute_weighted_means(S_i, λ, table_A, m_A, table_B, m_B)
    n_A = length(table_A)
    n_B = length(table_B)
    n_Si = length(S_i)
    λ_bar_Si = mean(view(λ, S_i))

    # New mean for augmented cluster B
    m_B_new = (n_B * m_B + n_Si * λ_bar_Si) / (n_B + n_Si)

    # New mean for depleted cluster A
    remaining_A = setdiff(table_A, S_i)
    if isempty(remaining_A)
        m_A_new = m_A
        log_jacobian = 0.0
    else
        n_remaining = length(remaining_A)
        m_A_new = (n_A * m_A - n_Si * λ_bar_Si) / n_remaining

        if m_A_new <= 0
            m_A_new = m_A
            log_jacobian = -Inf
        else
            log_jacobian = 0.0
        end
    end

    return m_A_new, m_B_new, log_jacobian
end

"""
    resample_posterior_means(S_i, λ, table_A, m_A, table_B, m_B, priors)

Resample means from approximate conditional posteriors.
"""
function resample_posterior_means(S_i, λ, table_A, m_A, table_B, m_B, priors)
    remaining_A = setdiff(table_A, S_i)
    augmented_B = vcat(table_B, S_i)

    # Sample new m_A from approximate posterior
    if isempty(remaining_A)
        m_A_new = m_A
        log_q_A_forward = 0.0
        log_q_A_reverse = 0.0
    else
        n_A_new = length(remaining_A)
        sum_λ_A = sum(view(λ, remaining_A))
        α_A = n_A_new + priors.m_a
        β_A = sum_λ_A + priors.m_b
        Q_A = InverseGamma(α_A, β_A)
        m_A_new = rand(Q_A)
        log_q_A_forward = logpdf(Q_A, m_A_new)

        α_A_rev = length(table_A) + priors.m_a
        β_A_rev = sum(view(λ, table_A)) + priors.m_b
        log_q_A_reverse = logpdf(InverseGamma(α_A_rev, β_A_rev), m_A)
    end

    # Sample new m_B from approximate posterior
    n_B_new = length(augmented_B)
    sum_λ_B = sum(view(λ, augmented_B))
    α_B = n_B_new + priors.m_a
    β_B = sum_λ_B + priors.m_b
    Q_B = InverseGamma(α_B, β_B)
    m_B_new = rand(Q_B)
    log_q_B_forward = logpdf(Q_B, m_B_new)

    α_B_rev = length(table_B) + priors.m_a
    β_B_rev = sum(view(λ, table_B)) + priors.m_b
    log_q_B_reverse = logpdf(InverseGamma(α_B_rev, β_B_rev), m_B)

    # Log proposal ratio: log q(reverse) - log q(forward)
    lpr = (log_q_A_reverse + log_q_B_reverse) - (log_q_A_forward + log_q_B_forward)

    return m_A_new, m_B_new, lpr
end

# ============================================================================
# Generic RJMCMC for Customer Assignments
# Uses the interface methods from rjmcmc_interface.jl:
#   cluster_param_dicts, copy_cluster_param_dicts, make_candidate_state,
#   commit_params!, sample_birth_params, birth_params_logpdf, fixed_dim_params
# ============================================================================

"""
    update_c_rjmcmc!(model, i, state, data, priors, proposal, log_DDCRP, opts)

Generic RJMCMC update for customer i's assignment.
Dispatches on interface methods to handle any number of cluster parameter dicts.

Returns (move_type::Symbol, j_star::Int, accepted::Bool)
"""
function update_c_rjmcmc!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    proposal::BirthProposal,
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    n = length(state.c)
    j_old = state.c[i]

    S_i = get_moving_set(i, state.c)
    dicts = cluster_param_dicts(state)
    primary_dict = first(dicts)
    table_Si = find_table_for_customer(i, primary_dict)
    table_l = setdiff(table_Si, S_i)

    j_star = rand(1:n)

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    params_can = copy_cluster_param_dicts(state)
    c_can = copy(state.c)
    c_can[i] = j_star

    if !j_old_in_Si && j_star_in_Si
        # ===== BIRTH MOVE =====
        params_new, log_q_forward = sample_birth_params(model, proposal, S_i, state, data, priors)

        sorted_Si = sort(S_i)
        for k in keys(params_can)
            params_can[k][sorted_Si] = params_new[k]
            if !isempty(table_l)
                params_can[k][sort(table_l)] = dicts[k][table_Si]
            end
            delete!(params_can[k], table_Si)
        end

        lpr = -log_q_forward

        state_can = make_candidate_state(model, state, c_can, params_can)
        log_α = posterior(model, data, state_can, priors, log_DDCRP) -
                posterior(model, data, state, priors, log_DDCRP) + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            commit_params!(state, params_can)
            return (:birth, j_star, true)
        end
        return (:birth, j_star, false)

    elseif j_old_in_Si && !j_star_in_Si
        # ===== DEATH MOVE =====
        table_target = find_table_for_customer(j_star, primary_dict)
        merged_table = sort(vcat(table_Si, table_target))

        for k in keys(params_can)
            params_can[k][merged_table] = dicts[k][table_target]
            delete!(params_can[k], table_Si)
            delete!(params_can[k], table_target)
        end

        # Old params for the moving set (for reverse proposal density)
        params_old = map(d -> d[table_Si], dicts)
        log_q_reverse = birth_params_logpdf(model, proposal, params_old, S_i, state, data, priors)
        lpr = log_q_reverse

        state_can = make_candidate_state(model, state, c_can, params_can)
        log_α = posterior(model, data, state_can, priors, log_DDCRP) -
                posterior(model, data, state, priors, log_DDCRP) + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            commit_params!(state, params_can)
            return (:death, j_star, true)
        end
        return (:death, j_star, false)

    else
        # ===== FIXED DIMENSION MOVE =====
        table_old_target = find_table_for_customer(j_old, primary_dict)
        table_new_target = find_table_for_customer(j_star, primary_dict)

        if table_old_target == table_new_target
            # Same table - just change link, no parameter changes
            state_can = make_candidate_state(model, state, c_can, dicts)
            log_α = posterior(model, data, state_can, priors, log_DDCRP) -
                    posterior(model, data, state, priors, log_DDCRP)

            if log(rand()) < log_α
                state.c[i] = j_star
                return (:fixed, j_star, true)
            end
            return (:fixed, j_star, false)
        end

        # Different tables - transfer S_i from old to new table
        new_table_depleted = setdiff(table_old_target, S_i)
        new_table_augmented = sort(vcat(table_new_target, S_i))

        params_depl, params_aug, lpr = fixed_dim_params(
            model, S_i, table_old_target, table_new_target, state, data, priors, opts)

        for k in keys(params_can)
            if !isempty(new_table_depleted)
                params_can[k][new_table_depleted] = params_depl[k]
            end
            params_can[k][new_table_augmented] = params_aug[k]
            delete!(params_can[k], table_old_target)
            delete!(params_can[k], table_new_target)
        end

        state_can = make_candidate_state(model, state, c_can, params_can)
        log_α = posterior(model, data, state_can, priors, log_DDCRP) -
                posterior(model, data, state, priors, log_DDCRP) + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            commit_params!(state, params_can)
            return (:fixed, j_star, true)
        end
        return (:fixed, j_star, false)
    end
end

# ============================================================================
# Cached RJMCMC: In-place modification with delta-posterior
# Eliminates dict/vector copies and computes only affected table contributions
# ============================================================================

"""
    update_c_rjmcmc_cached!(model, i, state, data, priors, proposal, log_DDCRP, opts)

Optimized RJMCMC update for customer i's assignment.
Uses in-place state modification with save/restore instead of copying,
and delta-posterior computation instead of full posterior evaluation.

Returns (move_type::Symbol, j_star::Int, accepted::Bool)
"""
function update_c_rjmcmc_cached!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    proposal::BirthProposal,
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    n = length(state.c)
    j_old = state.c[i]

    S_i = get_moving_set(i, state.c)
    dicts = cluster_param_dicts(state)
    primary_dict = first(dicts)
    table_Si = find_table_for_customer(i, primary_dict)
    table_l = setdiff(table_Si, S_i)

    j_star = rand(1:n)

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    # Delta DDCRP: only c[i] changes, so delta is O(1)
    ddcrp_delta = log_DDCRP[i, j_star] - log_DDCRP[i, j_old]

    if !j_old_in_Si && j_star_in_Si
        # ===== BIRTH MOVE =====
        # Sample birth params before modifying state
        params_new, log_q_forward = sample_birth_params(model, proposal, S_i, state, data, priors)

        sorted_Si = sort(S_i)
        sorted_table_l = isempty(table_l) ? Int[] : sort(table_l)

        # Compute old table contribution before modification
        old_contrib = table_contribution(model, sort(table_Si), state, data, priors)

        # Save affected entries
        saved = save_entries(dicts, [table_Si])
        old_table_Si_vals = map(d -> d[table_Si], dicts)

        # Modify state in-place
        state.c[i] = j_star
        for k in keys(dicts)
            dicts[k][sorted_Si] = params_new[k]
            if !isempty(sorted_table_l)
                dicts[k][sorted_table_l] = old_table_Si_vals[k]
            end
            delete!(dicts[k], table_Si)
        end

        # Compute new table contributions
        new_contrib = table_contribution(model, sorted_Si, state, data, priors)
        if !isempty(sorted_table_l)
            new_contrib += table_contribution(model, sorted_table_l, state, data, priors)
        end

        lpr = -log_q_forward
        log_α = (new_contrib - old_contrib) + ddcrp_delta + lpr

        if log(rand()) < log_α
            return (:birth, j_star, true)
        else
            # Restore state
            state.c[i] = j_old
            keys_to_delete = isempty(sorted_table_l) ? [sorted_Si] : [sorted_Si, sorted_table_l]
            restore_entries!(dicts, saved, keys_to_delete)
            return (:birth, j_star, false)
        end

    elseif j_old_in_Si && !j_star_in_Si
        # ===== DEATH MOVE =====
        table_target = find_table_for_customer(j_star, primary_dict)
        merged_table = sort(vcat(table_Si, table_target))

        # Reverse proposal density (before modification)
        params_old = map(d -> d[table_Si], dicts)
        log_q_reverse = birth_params_logpdf(model, proposal, params_old, S_i, state, data, priors)
        lpr = log_q_reverse

        # Compute old contributions before modification
        old_contrib = table_contribution(model, sort(table_Si), state, data, priors) +
                      table_contribution(model, sort(table_target), state, data, priors)

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
        new_table_depleted = setdiff(table_old_target, S_i)
        new_table_augmented = sort(vcat(table_new_target, S_i))

        # Compute fixed-dim params before modification
        params_depl, params_aug, lpr = fixed_dim_params(
            model, S_i, table_old_target, table_new_target, state, data, priors, opts)

        # Compute old contributions before modification
        old_contrib = table_contribution(model, sort(table_old_target), state, data, priors) +
                      table_contribution(model, sort(table_new_target), state, data, priors)

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
