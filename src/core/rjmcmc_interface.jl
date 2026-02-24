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
    fixed_dim_param(model, ::Val{name}, proposal, S_i, table_depl, table_aug, state, data, priors)
        -> (val_depleted, val_augmented, log_proposal_ratio::Float64)

Compute updated values for a **single** cluster parameter during a fixed-dimension move
where the moving set S_i transfers from `table_depl` to `table_aug`.

Returns the new parameter values for the depleted and augmented tables, plus the log
proposal ratio `log q(reverse) - log q(forward)`. For deterministic updates (NoUpdate,
WeightedMean) this ratio is 0.0. For stochastic updates (Resample) it is non-zero.

Dispatches on model type, `Val{name}` (the parameter name), and proposal type.
Models can override for model-specific behaviour (e.g., using latent variables instead
of raw observations for weighted-mean updates).
"""
function fixed_dim_param end

"""
    fixed_dim_params(model, proposal::FixedDimensionProposal,
                     S_i, table_depl, table_aug, state, data, priors)
        -> (params_depleted::NamedTuple, params_augmented::NamedTuple, log_proposal_ratio::Float64)

Compute updated parameter values for a different-table fixed-dimension move.
Dispatches per-parameter to `fixed_dim_param` for each key in `cluster_param_dicts`.
Returns scalar NamedTuples for the depleted and augmented tables, plus total lpr.
"""
function fixed_dim_params(model::LikelihoodModel, proposal::FixedDimensionProposal,
                          S_i::Vector{Int}, table_depl::Vector{Int}, table_aug::Vector{Int},
                          state::AbstractMCMCState, data::AbstractObservedData,
                          priors::AbstractPriors)
    dicts = cluster_param_dicts(state)
    names = keys(dicts)
    depl_vals = Vector{Any}(undef, length(names))
    aug_vals  = Vector{Any}(undef, length(names))
    total_lpr = 0.0
    for (idx, name) in enumerate(names)
        vd, va, lpr_k = fixed_dim_param(model, Val(name), proposal, S_i, table_depl, table_aug, state, data, priors)
        depl_vals[idx] = vd
        aug_vals[idx]  = va
        total_lpr += lpr_k
    end
    return NamedTuple{names}(depl_vals), NamedTuple{names}(aug_vals), total_lpr
end

"""
    fixed_dim_params(model, proposal::MixedFixedDim, ...)

MixedFixedDim override: dispatches each parameter to its own per-parameter proposal.
Parameters not present in `proposal.proposals` fall back to `NoUpdate`.
"""
function fixed_dim_params(model::LikelihoodModel, proposal::MixedFixedDim,
                          S_i::Vector{Int}, table_depl::Vector{Int}, table_aug::Vector{Int},
                          state::AbstractMCMCState, data::AbstractObservedData,
                          priors::AbstractPriors)
    dicts = cluster_param_dicts(state)
    names = keys(dicts)
    depl_vals = Vector{Any}(undef, length(names))
    aug_vals  = Vector{Any}(undef, length(names))
    total_lpr = 0.0
    for (idx, name) in enumerate(names)
        p = get(proposal.proposals, name, NoUpdate())
        vd, va, lpr_k = fixed_dim_param(model, Val(name), p, S_i, table_depl, table_aug, state, data, priors)
        depl_vals[idx] = vd
        aug_vals[idx]  = va
        total_lpr += lpr_k
    end
    return NamedTuple{names}(depl_vals), NamedTuple{names}(aug_vals), total_lpr
end

# ============================================================================
# Generic fixed_dim_param implementations (work for all models)
# ============================================================================

"""
    fixed_dim_param(model, ::Val{name}, ::NoUpdate, ...)

Keep existing cluster parameters unchanged. Always returns lpr = 0.0.
"""
function fixed_dim_param(_model::LikelihoodModel, ::Val{name}, ::NoUpdate,
                         _S_i::Vector{Int}, table_depl::Vector{Int}, table_aug::Vector{Int},
                         state::AbstractMCMCState, _data::AbstractObservedData,
                         _priors::AbstractPriors) where {name}
    dicts = cluster_param_dicts(state)
    return dicts[name][table_depl], dicts[name][table_aug], 0.0
end

"""
    fixed_dim_param(model, ::Val{name}, ::WeightedMean, ...)

Deterministically update a parameter as a weighted average.
Uses the mean of raw observations in S_i as the summary statistic.
Returns lpr = 0.0 (deterministic, unit Jacobian). Returns lpr = -Inf to
force rejection if the depleted parameter would become non-positive.

Models can override this for parameters where a different summary statistic
is appropriate (e.g., using latent variables instead of raw observations).
"""
function fixed_dim_param(_model::LikelihoodModel, ::Val{name}, ::WeightedMean,
                         S_i::Vector{Int}, table_depl::Vector{Int}, table_aug::Vector{Int},
                         state::AbstractMCMCState, data::AbstractObservedData,
                         _priors::AbstractPriors) where {name}
    dicts = cluster_param_dicts(state)
    param_depl = dicts[name][table_depl]
    param_aug  = dicts[name][table_aug]

    ȳ_Si = mean(view(observations(data), S_i))
    n_depl = length(table_depl)
    n_aug  = length(table_aug)
    n_Si   = length(S_i)

    param_aug_new = (n_aug * param_aug + n_Si * ȳ_Si) / (n_aug + n_Si)

    n_remaining = n_depl - n_Si
    if n_remaining > 0
        param_depl_new = (n_depl * param_depl - n_Si * ȳ_Si) / n_remaining
        lpr = param_depl_new <= 0 ? -Inf : 0.0
    else
        param_depl_new = param_depl  # depleted cluster will be empty; value unused
        lpr = 0.0
    end

    return param_depl_new, param_aug_new, lpr
end

"""
    fixed_dim_param(model, vname::Val{name}, proposal::Resample, ...)

Stochastically resample a parameter for the modified clusters by reusing the
inner birth proposal. Calls `sample_birth_param`/`birth_param_logpdf` on the
new cluster memberships (remaining depleted and augmented sets).
"""
function fixed_dim_param(model::LikelihoodModel, vname::Val{name}, proposal::Resample,
                         S_i::Vector{Int}, table_depl::Vector{Int}, table_aug::Vector{Int},
                         state::AbstractMCMCState, data::AbstractObservedData,
                         priors::AbstractPriors) where {name}
    dicts = cluster_param_dicts(state)
    inner = proposal.proposal

    remaining_depl = sorted_setdiff(table_depl, S_i)
    augmented_aug  = sorted_merge(table_aug, S_i)

    # Augmented cluster: always resample
    val_aug_new, lq_fwd_aug = sample_birth_param(model, vname, inner, augmented_aug, state, data, priors)
    val_aug_old = dicts[name][table_aug]
    lq_rev_aug  = birth_param_logpdf(model, vname, inner, val_aug_old, augmented_aug, state, data, priors)

    # Depleted cluster: resample only if non-empty, otherwise keep value
    if !isempty(remaining_depl)
        val_depl_new, lq_fwd_depl = sample_birth_param(model, vname, inner, remaining_depl, state, data, priors)
        val_depl_old = dicts[name][table_depl]
        lq_rev_depl  = birth_param_logpdf(model, vname, inner, val_depl_old, remaining_depl, state, data, priors)
    else
        val_depl_new = dicts[name][table_depl]
        lq_fwd_depl  = 0.0
        lq_rev_depl  = 0.0
    end

    lpr = (lq_rev_depl + lq_rev_aug) - (lq_fwd_depl + lq_fwd_aug)
    return val_depl_new, val_aug_new, lpr
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
