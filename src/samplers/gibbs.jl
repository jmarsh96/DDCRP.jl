# ============================================================================
# Gibbs Sampling for Customer Assignments
# Used with GibbsProposal for marginalised models
# ============================================================================

using StatsBase

"""
    update_c!(::GibbsProposal, model, i, state, y, priors, log_DDCRP)

Update customer i's assignment using Gibbs sampling.
Computes probabilities for linking to each customer based on table contributions.

This generic implementation works for any marginalised model that implements
`table_contribution(model, table, state, y, priors)`.

Returns (move_type, new_assignment, accepted) for consistency with RJMCMC interface.
For Gibbs sampling, move_type is always :gibbs and accepted is always true.
"""
function update_c!(
    ::GibbsProposal,
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    y::AbstractVector,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix
)
    n = length(state.c)
    log_probs = zeros(n)

    # Get tables with customer i's link temporarily removed
    table_minus_i = table_vector_minus_i(i, state.c)
    current_table = table_minus_i[findfirst(x -> i in x, table_minus_i)]

    # Compute contribution from current table
    current_table_contrib = table_contribution(model, current_table, state, y, priors)

    # Probability for linking to customers in current table
    for customer in current_table
        log_probs[customer] = log_DDCRP[i, customer]
    end

    # Probability for linking to customers in other tables
    remaining_tables = setdiff(table_minus_i, [current_table])

    for table in remaining_tables
        prop_table_contrib = table_contribution(model, table, state, y, priors)
        joined_table = vcat(current_table, table)
        joined_table_contrib = table_contribution(model, joined_table, state, y, priors)

        # Log ratio for joining this table
        term = joined_table_contrib - prop_table_contrib - current_table_contrib

        for customer in table
            log_probs[customer] = log_DDCRP[i, customer] + term
        end
    end

    # Sample new assignment
    probs = exp.(log_probs .- maximum(log_probs))
    new_assignment = sample(1:n, Weights(probs))
    state.c[i] = new_assignment

    return (:gibbs, new_assignment, true)
end

# ============================================================================
# Specialized implementations for specific model types
# ============================================================================

# NBGammaPoissonGlobalRMarg - uses state-only table_contribution
function update_c!(
    ::GibbsProposal,
    model::NBGammaPoissonGlobalRMarg,
    i::Int,
    state::NBGammaPoissonGlobalRMargState,
    y::AbstractVector,
    priors::NBGammaPoissonGlobalRMargPriors,
    log_DDCRP::AbstractMatrix
)
    n = length(state.c)
    log_probs = zeros(n)

    table_minus_i = table_vector_minus_i(i, state.c)
    current_table = table_minus_i[findfirst(x -> i in x, table_minus_i)]

    # NBGammaPoissonGlobalRMarg uses table_contribution(model, table, state, priors)
    current_table_contrib = table_contribution(model, current_table, state, priors)

    for customer in current_table
        log_probs[customer] = log_DDCRP[i, customer]
    end

    remaining_tables = setdiff(table_minus_i, [current_table])

    for table in remaining_tables
        prop_table_contrib = table_contribution(model, table, state, priors)
        joined_table = vcat(current_table, table)
        joined_table_contrib = table_contribution(model, joined_table, state, priors)

        term = joined_table_contrib - prop_table_contrib - current_table_contrib

        for customer in table
            log_probs[customer] = log_DDCRP[i, customer] + term
        end
    end

    probs = exp.(log_probs .- maximum(log_probs))
    state.c[i] = sample(1:n, Weights(probs))

    return (:gibbs, state.c[i], true)
end

# PoissonClusterRatesMarg - passes y to table_contribution
function update_c!(
    ::GibbsProposal,
    model::PoissonClusterRatesMarg,
    i::Int,
    state::PoissonClusterRatesMargState,
    y::AbstractVector,
    priors::PoissonClusterRatesMargPriors,
    log_DDCRP::AbstractMatrix
)
    n = length(state.c)
    log_probs = zeros(n)

    table_minus_i = table_vector_minus_i(i, state.c)
    current_table = table_minus_i[findfirst(x -> i in x, table_minus_i)]

    current_table_contrib = table_contribution(model, current_table, state, y, priors)

    for customer in current_table
        log_probs[customer] = log_DDCRP[i, customer]
    end

    remaining_tables = setdiff(table_minus_i, [current_table])

    for table in remaining_tables
        prop_table_contrib = table_contribution(model, table, state, y, priors)
        joined_table = vcat(current_table, table)
        joined_table_contrib = table_contribution(model, joined_table, state, y, priors)

        term = joined_table_contrib - prop_table_contrib - current_table_contrib

        for customer in table
            log_probs[customer] = log_DDCRP[i, customer] + term
        end
    end

    probs = exp.(log_probs .- maximum(log_probs))
    state.c[i] = sample(1:n, Weights(probs))

    return (:gibbs, state.c[i], true)
end

# BinomialClusterProbMarg - passes y, N to table_contribution
function update_c!(
    ::GibbsProposal,
    model::BinomialClusterProbMarg,
    i::Int,
    state::BinomialClusterProbMargState,
    y::AbstractVector,
    N::Union{Int, AbstractVector{Int}},
    priors::BinomialClusterProbMargPriors,
    log_DDCRP::AbstractMatrix
)
    n = length(state.c)
    log_probs = zeros(n)

    table_minus_i = table_vector_minus_i(i, state.c)
    current_table = table_minus_i[findfirst(x -> i in x, table_minus_i)]

    current_table_contrib = table_contribution(model, current_table, state, y, N, priors)

    for customer in current_table
        log_probs[customer] = log_DDCRP[i, customer]
    end

    remaining_tables = setdiff(table_minus_i, [current_table])

    for table in remaining_tables
        prop_table_contrib = table_contribution(model, table, state, y, N, priors)
        joined_table = vcat(current_table, table)
        joined_table_contrib = table_contribution(model, joined_table, state, y, N, priors)

        term = joined_table_contrib - prop_table_contrib - current_table_contrib

        for customer in table
            log_probs[customer] = log_DDCRP[i, customer] + term
        end
    end

    probs = exp.(log_probs .- maximum(log_probs))
    state.c[i] = sample(1:n, Weights(probs))

    return (:gibbs, state.c[i], true)
end

# ============================================================================
# MetropolisProposal - Simple MH for unmarginalised models
# ============================================================================

"""
    update_c!(::MetropolisProposal, model, i, state, y, priors, log_DDCRP)

Update customer i's assignment using Metropolis-Hastings with uniform proposal.
Proposes a random link and accepts/rejects based on posterior ratio.
"""
function update_c!(
    ::MetropolisProposal,
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    y::AbstractVector,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix
)
    n = length(state.c)
    j_old = state.c[i]

    # Propose new link uniformly
    j_star = rand(1:n)

    # Create candidate state
    c_can = copy(state.c)
    c_can[i] = j_star

    # Compute acceptance ratio (symmetric proposal, so just posterior ratio)
    # This requires creating a temporary state - model specific
    log_α = log_DDCRP[i, j_star] - log_DDCRP[i, j_old]

    if log(rand()) < log_α
        state.c[i] = j_star
        return (:metropolis, j_star, true)
    end
    return (:metropolis, j_star, false)
end

# ============================================================================
# Legacy compatibility - old signature with strategy first
# ============================================================================

# These allow old code using update_c!(strategy, model, ...) to still work

function update_c!(
    strategy::MarginalisedStrategy,
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    y::AbstractVector,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix;
    opts = nothing
)
    return update_c!(GibbsProposal(), model, i, state, y, priors, log_DDCRP)
end
