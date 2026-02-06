# ============================================================================
# Gibbs Sampling for Customer Assignments
# ============================================================================

using StatsBase

# ============================================================================
# Internal Gibbs update functions (called by model's update_params!)
# ============================================================================

"""
    update_c_gibbs!(model, i, state, data, priors, log_DDCRP)

Internal Gibbs sampling update for customer i's assignment.
Called by `update_c!` for marginalised models or ConjugateProposal.

Returns (move_type, new_assignment, accepted).
For Gibbs sampling, move_type is always :gibbs and accepted is always true.
"""
function update_c_gibbs!(
    model::LikelihoodModel,
    i::Int,
    state::AbstractMCMCState,
    data::AbstractObservedData,
    priors::AbstractPriors,
    log_DDCRP::AbstractMatrix
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

# NBGammaPoissonGlobalRMarg - uses state-only table_contribution
function update_c_gibbs!(
    model::NBGammaPoissonGlobalRMarg,
    i::Int,
    state::NBGammaPoissonGlobalRMargState,
    data::CountData,
    priors::NBGammaPoissonGlobalRMargPriors,
    log_DDCRP::AbstractMatrix
)
    n = length(state.c)
    log_probs = zeros(n)

    table_minus_i = table_vector_minus_i(i, state.c)
    current_table = table_minus_i[findfirst(x -> i in x, table_minus_i)]

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

# PoissonClusterRatesMarg - passes data to table_contribution
function update_c_gibbs!(
    model::PoissonClusterRatesMarg,
    i::Int,
    state::PoissonClusterRatesMargState,
    data::CountData,
    priors::PoissonClusterRatesMargPriors,
    log_DDCRP::AbstractMatrix
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
    state.c[i] = sample(1:n, Weights(probs))

    return (:gibbs, state.c[i], true)
end

# BinomialClusterProbMarg - requires CountDataWithTrials for table_contribution
function update_c_gibbs!(
    model::BinomialClusterProbMarg,
    i::Int,
    state::BinomialClusterProbMargState,
    data::CountDataWithTrials,
    priors::BinomialClusterProbMargPriors,
    log_DDCRP::AbstractMatrix
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
    state.c[i] = sample(1:n, Weights(probs))

    return (:gibbs, state.c[i], true)
end

