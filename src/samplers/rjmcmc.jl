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
    return nothing
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
# RJMCMC Customer Assignment Update - New Signature
# ============================================================================

"""
    update_c!(proposal::RJMCMCProposal, model, i, state, y, priors, log_DDCRP)

Update customer i's assignment using Reversible Jump MCMC.

Move types:
- Birth (K→K+1): S_i splits off to form new cluster
- Death (K→K-1): S_i merges into existing cluster
- Fixed dimension: transfer between clusters or internal rewiring

Returns (move_type::Symbol, j_star::Int, accepted::Bool)
"""
function update_c!(
    proposal::RJMCMCProposal,
    model::NBGammaPoissonGlobalR,
    i::Int,
    state::NBGammaPoissonGlobalRState,
    y::AbstractVector,
    priors::NBGammaPoissonGlobalRPriors,
    log_DDCRP::AbstractMatrix
)
    n = length(state.c)
    j_old = state.c[i]

    S_i = get_moving_set(i, state.c)
    table_Si = find_table_for_customer(i, state.m_dict)
    m_old = state.m_dict[table_Si]
    table_l = setdiff(table_Si, S_i)

    # Propose new link uniformly
    j_star = rand(1:n)

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    m_can = copy(state.m_dict)
    c_can = copy(state.c)
    c_can[i] = j_star

    birth_prop = proposal.birth_proposal
    fixed_dim_mode = proposal.fixed_dim_mode

    if !j_old_in_Si && j_star_in_Si
        # BIRTH: S_i splits off to form new cluster
        m_new, log_q_forward = sample_proposal(birth_prop, S_i, state.λ, priors)

        m_can[sort(S_i)] = m_new
        m_can[sort(table_l)] = state.m_dict[table_Si]
        delete!(m_can, table_Si)

        # Hastings ratio: -log q(m_new | forward)
        lpr = -log_q_forward

        state_can = NBGammaPoissonGlobalRState(c_can, state.λ, m_can, state.r)
        log_α = posterior(model, y, state_can, priors, log_DDCRP) -
                posterior(model, y, state, priors, log_DDCRP) + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.m_dict)
            merge!(state.m_dict, m_can)
            return (:birth, j_star, true)
        end
        return (:birth, j_star, false)

    elseif j_old_in_Si && !j_star_in_Si
        # DEATH: S_i merges into another cluster
        table_target = find_table_for_customer(j_star, state.m_dict)
        m_target = state.m_dict[table_target]

        m_can[sort(vcat(table_Si, table_target))] = m_target

        # Hastings ratio: +log q(m_old | reverse birth)
        m_old = state.m_dict[table_Si]
        log_q_reverse = proposal_logpdf(birth_prop, m_old, S_i, state.λ, priors)
        lpr = log_q_reverse

        delete!(m_can, table_Si)
        delete!(m_can, table_target)

        state_can = NBGammaPoissonGlobalRState(c_can, state.λ, m_can, state.r)
        log_α = posterior(model, y, state_can, priors, log_DDCRP) -
                posterior(model, y, state, priors, log_DDCRP) + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.m_dict)
            merge!(state.m_dict, m_can)
            return (:death, j_star, true)
        end
        return (:death, j_star, false)

    else
        # FIXED DIMENSION: both inside S_i or both outside
        table_old_target = find_table_for_customer(j_old, state.m_dict)
        table_new_target = find_table_for_customer(j_star, state.m_dict)

        if table_old_target == table_new_target
            # Same cluster - internal rewiring only
            state_can = NBGammaPoissonGlobalRState(c_can, state.λ, state.m_dict, state.r)

            log_α = posterior(model, y, state_can, priors, log_DDCRP) -
                    posterior(model, y, state, priors, log_DDCRP)

            if log(rand()) < log_α
                state.c[i] = j_star
                return (:fixed, j_star, true)
            end
            return (:fixed, j_star, false)
        end

        # Different clusters - S_i moves from table_old_target to table_new_target
        new_table_depleted = setdiff(table_old_target, S_i)
        new_table_augmented = sort(vcat(table_new_target, S_i))

        # Compute new means based on fixed_dim_mode
        m_depleted, m_augmented, lpr = compute_fixed_dim_means(
            fixed_dim_mode, S_i, state.λ,
            table_old_target, state.m_dict[table_old_target],
            table_new_target, state.m_dict[table_new_target],
            priors
        )

        m_can[new_table_depleted] = m_depleted
        m_can[new_table_augmented] = m_augmented
        delete!(m_can, table_old_target)
        delete!(m_can, table_new_target)

        state_can = NBGammaPoissonGlobalRState(c_can, state.λ, m_can, state.r)
        log_α = posterior(model, y, state_can, priors, log_DDCRP) -
                posterior(model, y, state, priors, log_DDCRP) + lpr

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.m_dict)
            merge!(state.m_dict, m_can)
            return (:fixed, j_star, true)
        end
        return (:fixed, j_star, false)
    end
end

# ============================================================================
# RJMCMC for PoissonClusterRates
# ============================================================================

function update_c!(
    proposal::RJMCMCProposal,
    model::PoissonClusterRates,
    i::Int,
    state::PoissonClusterRatesState,
    y::AbstractVector,
    priors::PoissonClusterRatesPriors,
    log_DDCRP::AbstractMatrix
)
    n = length(state.c)
    j_old = state.c[i]

    S_i = get_moving_set(i, state.c)
    table_Si = find_table_for_customer(i, state.λ_dict)
    λ_old = state.λ_dict[table_Si]
    table_l = setdiff(table_Si, S_i)

    j_star = rand(1:n)

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    λ_can = copy(state.λ_dict)
    c_can = copy(state.c)
    c_can[i] = j_star

    if !j_old_in_Si && j_star_in_Si
        # BIRTH: Sample new rate from posterior
        S_k = sum(view(y, S_i))
        n_k = length(S_i)
        λ_new = rand(Gamma(priors.λ_a + S_k, 1/(priors.λ_b + n_k)))

        λ_can[sort(S_i)] = λ_new
        λ_can[sort(table_l)] = state.λ_dict[table_Si]
        delete!(λ_can, table_Si)

        state_can = PoissonClusterRatesState(c_can, λ_can)
        log_α = posterior(model, y, state_can, priors, log_DDCRP) -
                posterior(model, y, state, priors, log_DDCRP)

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.λ_dict)
            merge!(state.λ_dict, λ_can)
            return (:birth, j_star, true)
        end
        return (:birth, j_star, false)

    elseif j_old_in_Si && !j_star_in_Si
        # DEATH: Merge into target cluster
        table_target = find_table_for_customer(j_star, state.λ_dict)
        λ_target = state.λ_dict[table_target]

        λ_can[sort(vcat(table_Si, table_target))] = λ_target
        delete!(λ_can, table_Si)
        delete!(λ_can, table_target)

        state_can = PoissonClusterRatesState(c_can, λ_can)
        log_α = posterior(model, y, state_can, priors, log_DDCRP) -
                posterior(model, y, state, priors, log_DDCRP)

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.λ_dict)
            merge!(state.λ_dict, λ_can)
            return (:death, j_star, true)
        end
        return (:death, j_star, false)

    else
        # FIXED: Internal rewiring or cluster transfer
        table_old_target = find_table_for_customer(j_old, state.λ_dict)
        table_new_target = find_table_for_customer(j_star, state.λ_dict)

        if table_old_target == table_new_target
            state_can = PoissonClusterRatesState(c_can, state.λ_dict)
            log_α = posterior(model, y, state_can, priors, log_DDCRP) -
                    posterior(model, y, state, priors, log_DDCRP)

            if log(rand()) < log_α
                state.c[i] = j_star
                return (:fixed, j_star, true)
            end
            return (:fixed, j_star, false)
        end

        # Transfer between clusters
        new_table_depleted = setdiff(table_old_target, S_i)
        new_table_augmented = sort(vcat(table_new_target, S_i))

        λ_can[new_table_depleted] = state.λ_dict[table_old_target]
        λ_can[new_table_augmented] = state.λ_dict[table_new_target]
        delete!(λ_can, table_old_target)
        delete!(λ_can, table_new_target)

        state_can = PoissonClusterRatesState(c_can, λ_can)
        log_α = posterior(model, y, state_can, priors, log_DDCRP) -
                posterior(model, y, state, priors, log_DDCRP)

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.λ_dict)
            merge!(state.λ_dict, λ_can)
            return (:fixed, j_star, true)
        end
        return (:fixed, j_star, false)
    end
end

# ============================================================================
# RJMCMC for BinomialClusterProb
# ============================================================================

function update_c!(
    proposal::RJMCMCProposal,
    model::BinomialClusterProb,
    i::Int,
    state::BinomialClusterProbState,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    priors::BinomialClusterProbPriors,
    log_DDCRP::AbstractMatrix
)
    n = length(state.c)
    j_old = state.c[i]

    S_i = get_moving_set(i, state.c)
    table_Si = find_table_for_customer(i, state.p_dict)
    p_old = state.p_dict[table_Si]
    table_l = setdiff(table_Si, S_i)

    j_star = rand(1:n)

    j_old_in_Si = j_old in S_i
    j_star_in_Si = j_star in S_i

    p_can = copy(state.p_dict)
    c_can = copy(state.c)
    c_can[i] = j_star

    if !j_old_in_Si && j_star_in_Si
        # BIRTH: Sample new probability from posterior
        S_k = sum(view(y, S_i))
        N_k = N isa Int ? N * length(S_i) : sum(view(N, S_i))
        p_new = rand(Beta(priors.p_a + S_k, priors.p_b + N_k - S_k))

        p_can[sort(S_i)] = p_new
        p_can[sort(table_l)] = state.p_dict[table_Si]
        delete!(p_can, table_Si)

        state_can = BinomialClusterProbState(c_can, p_can)
        log_α = posterior(model, y, N, state_can, priors, log_DDCRP) -
                posterior(model, y, N, state, priors, log_DDCRP)

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.p_dict)
            merge!(state.p_dict, p_can)
            return (:birth, j_star, true)
        end
        return (:birth, j_star, false)

    elseif j_old_in_Si && !j_star_in_Si
        # DEATH: Merge into target cluster
        table_target = find_table_for_customer(j_star, state.p_dict)
        p_target = state.p_dict[table_target]

        p_can[sort(vcat(table_Si, table_target))] = p_target
        delete!(p_can, table_Si)
        delete!(p_can, table_target)

        state_can = BinomialClusterProbState(c_can, p_can)
        log_α = posterior(model, y, N, state_can, priors, log_DDCRP) -
                posterior(model, y, N, state, priors, log_DDCRP)

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.p_dict)
            merge!(state.p_dict, p_can)
            return (:death, j_star, true)
        end
        return (:death, j_star, false)

    else
        # FIXED: Internal rewiring or cluster transfer
        table_old_target = find_table_for_customer(j_old, state.p_dict)
        table_new_target = find_table_for_customer(j_star, state.p_dict)

        if table_old_target == table_new_target
            state_can = BinomialClusterProbState(c_can, state.p_dict)
            log_α = posterior(model, y, N, state_can, priors, log_DDCRP) -
                    posterior(model, y, N, state, priors, log_DDCRP)

            if log(rand()) < log_α
                state.c[i] = j_star
                return (:fixed, j_star, true)
            end
            return (:fixed, j_star, false)
        end

        # Transfer between clusters
        new_table_depleted = setdiff(table_old_target, S_i)
        new_table_augmented = sort(vcat(table_new_target, S_i))

        p_can[new_table_depleted] = state.p_dict[table_old_target]
        p_can[new_table_augmented] = state.p_dict[table_new_target]
        delete!(p_can, table_old_target)
        delete!(p_can, table_new_target)

        state_can = BinomialClusterProbState(c_can, p_can)
        log_α = posterior(model, y, N, state_can, priors, log_DDCRP) -
                posterior(model, y, N, state, priors, log_DDCRP)

        if log(rand()) < log_α
            state.c[i] = j_star
            empty!(state.p_dict)
            merge!(state.p_dict, p_can)
            return (:fixed, j_star, true)
        end
        return (:fixed, j_star, false)
    end
end

# ============================================================================
# Legacy compatibility - old signature with RJMCMC strategy
# ============================================================================

function update_c!(
    strategy::RJMCMC_Strategy,
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    y::AbstractVector,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix;
    opts = nothing
)
    bp = isnothing(opts) ? PriorProposal() : get(opts, :birth_proposal, PriorProposal())
    fdm = isnothing(opts) ? :none : get(opts, :fixed_dim_mode, :none)
    proposal = RJMCMCProposal(bp, fdm)
    return update_c!(proposal, model, i, state, y, priors, log_DDCRP)
end
