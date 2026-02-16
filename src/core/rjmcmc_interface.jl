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
    copy_cluster_param_dicts(state::AbstractMCMCState) -> NamedTuple

Return shallow copies of all cluster parameter dicts.
Same keys as `cluster_param_dicts`.
"""
function copy_cluster_param_dicts end

"""
    make_candidate_state(model::LikelihoodModel, state::AbstractMCMCState,
                         c_can::Vector{Int}, params_can::NamedTuple)

Construct a candidate state from a candidate assignment vector and
candidate cluster parameter dicts. Non-dict fields (e.g., global r,
observation-level λ, latent h) are copied from the current state.
"""
function make_candidate_state end

"""
    commit_params!(state::AbstractMCMCState, params_can::NamedTuple)

Apply accepted candidate parameters to the current state.
For each dict: `empty!(state_dict); merge!(state_dict, can_dict)`.
"""
function commit_params! end

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
