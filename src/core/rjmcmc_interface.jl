# ============================================================================
# Generic RJMCMC Interface Methods
# Each unmarginalised model must implement these to use the generic
# update_c_rjmcmc! function.
# ============================================================================

"""
    cluster_param_dicts(state::AbstractMCMCState) -> NamedTuple

Return a NamedTuple of all cluster parameter dicts from the state.
The first dict is used as the "primary" dict for table lookups.

Each model returns its specific set of dicts, e.g.:
- `(m = state.m_dict,)` for 1-parameter models
- `(m = state.m_dict, r = state.r_dict)` for 2-parameter models
- `(ξ = state.ξ_dict, ω = state.ω_dict, α = state.α_dict)` for 3-parameter models
"""
function cluster_param_dicts end


"""
    sample_birth_params(model::LikelihoodModel, proposal::BirthProposal,
                        S_i::Vector{Int}, state::AbstractMCMCState,
                        data::AbstractObservedData, priors::AbstractPriors)
        -> (params_new::NamedTuple, log_q_forward::Float64)

Sample new cluster parameters for a birth move.
`params_new` has the same keys as `cluster_param_dicts` but with scalar values.

Dispatches on both model type and proposal type.
"""
function sample_birth_params end

"""
    birth_params_logpdf(model::LikelihoodModel, proposal::BirthProposal,
                        params_old::NamedTuple, S_i::Vector{Int},
                        state::AbstractMCMCState, data::AbstractObservedData,
                        priors::AbstractPriors) -> Float64

Log density of the birth proposal at the given parameter values.
Used in death moves to compute the reverse Hastings ratio.

Dispatches on both model type and proposal type.
"""
function birth_params_logpdf end

"""
    sample_birth_param(model::LikelihoodModel, ::Val{param_name},
                       proposal::BirthProposal, S_i::Vector{Int},
                       state::AbstractMCMCState, data::AbstractObservedData,
                       priors::AbstractPriors)
        -> (value, log_q_forward::Float64)

Sample a single cluster parameter for a birth move.
Used by `MixedProposal` to dispatch each parameter independently.

Dispatches on model type, `Val{param_name}`, and proposal type.
Each model implements this for the (parameter, proposal) combinations it supports.
"""
function sample_birth_param end

"""
    birth_param_logpdf(model::LikelihoodModel, ::Val{param_name},
                       proposal::BirthProposal, param_value,
                       S_i::Vector{Int}, state::AbstractMCMCState,
                       data::AbstractObservedData, priors::AbstractPriors)
        -> Float64

Log density of the per-parameter birth proposal at `param_value`.
Used by `MixedProposal` in death moves to compute the reverse Hastings ratio.

Dispatches on model type, `Val{param_name}`, and proposal type.
"""
function birth_param_logpdf end

"""
    fixed_dim_params(model::LikelihoodModel, S_i::Vector{Int},
                     table_old::Vector{Int}, table_new::Vector{Int},
                     state::AbstractMCMCState, data::AbstractObservedData,
                     priors::AbstractPriors, opts::MCMCOptions)
        -> (params_depleted::NamedTuple, params_augmented::NamedTuple, log_proposal_ratio::Float64)

Compute updated parameter values for a different-table fixed-dimension move.
Returns scalar NamedTuples for the depleted and augmented tables,
plus a log proposal ratio.

Default: keep existing parameters unchanged, lpr = 0.0.
"""
function fixed_dim_params(model::LikelihoodModel, S_i::Vector{Int},
                          table_old::Vector{Int}, table_new::Vector{Int},
                          state::AbstractMCMCState, data::AbstractObservedData,
                          priors::AbstractPriors, opts)
    dicts = cluster_param_dicts(state)
    params_depl = map(d -> d[table_old], dicts)
    params_aug = map(d -> d[table_new], dicts)
    return params_depl, params_aug, 0.0
end

# ============================================================================
# Generic Save/Restore Helpers for In-Place RJMCMC
# ============================================================================

"""
    save_entries(dicts::NamedTuple, table_keys)

Save current entries for the given table keys from each dict in the NamedTuple.
Returns a NamedTuple of `Vector{Pair{Vector{Int}, T}}` for each dict.
"""
function save_entries(dicts::NamedTuple, table_keys)
    return map(d -> [k => d[k] for k in table_keys if haskey(d, k)], dicts)
end

"""
    restore_entries!(dicts::NamedTuple, saved::NamedTuple, keys_to_delete)

Restore dicts to their saved state:
1. Delete any keys in `keys_to_delete` from all dicts
2. Re-insert all saved entries
"""
function restore_entries!(dicts::NamedTuple, saved::NamedTuple, keys_to_delete)
    for name in keys(dicts)
        for k in keys_to_delete
            if haskey(dicts[name], k)
                delete!(dicts[name], k)
            end
        end
        for (k, v) in saved[name]
            dicts[name][k] = v
        end
    end
end

# ============================================================================
# Sorted Vector Utilities (allocation-free alternatives to setdiff/vcat+sort)
# ============================================================================

"""
    sorted_setdiff(a::Vector{Int}, b::Vector{Int}) -> Vector{Int}

Compute setdiff(a, b) for sorted vectors. Returns a sorted result.
Avoids the Set/Dict allocation that Base.setdiff uses internally.
Both `a` and `b` must be sorted in ascending order.
"""
function sorted_setdiff(a::Vector{Int}, b::Vector{Int})
    result = Int[]
    ia, ib = 1, 1
    na, nb = length(a), length(b)
    @inbounds while ia <= na
        if ib > nb || a[ia] < b[ib]
            push!(result, a[ia])
            ia += 1
        elseif a[ia] == b[ib]
            ia += 1
            ib += 1
        else
            ib += 1
        end
    end
    return result
end

"""
    sorted_merge(a::Vector{Int}, b::Vector{Int}) -> Vector{Int}

Merge two sorted vectors into a single sorted vector.
Equivalent to `sort(vcat(a, b))` but avoids intermediate allocation and sorting.
Assumes no duplicates between `a` and `b` (disjoint sets).
"""
function sorted_merge(a::Vector{Int}, b::Vector{Int})
    na, nb = length(a), length(b)
    result = Vector{Int}(undef, na + nb)
    ia, ib, ir = 1, 1, 1
    @inbounds while ia <= na && ib <= nb
        if a[ia] <= b[ib]
            result[ir] = a[ia]
            ia += 1
        else
            result[ir] = b[ib]
            ib += 1
        end
        ir += 1
    end
    @inbounds while ia <= na
        result[ir] = a[ia]
        ia += 1
        ir += 1
    end
    @inbounds while ib <= nb
        result[ir] = b[ib]
        ib += 1
        ir += 1
    end
    return result
end
