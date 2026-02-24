# ============================================================================
# MCMCOptions - Configuration for MCMC sampling
# ============================================================================

"""
    MCMCOptions

Configuration for MCMC sampling. Birth proposals and fixed-dimension proposals
are passed directly to `mcmc` as arguments, not through options.

# Fields
- `n_samples::Int`: Number of MCMC iterations (default: 10000)
- `verbose::Bool`: Print progress (default: false)
- `infer_params::Dict{Symbol, Bool}`: Which parameters to infer (default: all true)
- `prop_sds::Dict{Symbol, Float64}`: Proposal standard deviations for MH parameter updates
- `track_diagnostics::Bool`: Track acceptance rates (default: true)
- `track_pairwise::Bool`: Track pairwise proposals (default: false)
"""
struct MCMCOptions
    n_samples::Int
    verbose::Bool
    infer_params::Dict{Symbol, Bool}
    prop_sds::Dict{Symbol, Float64}
    track_diagnostics::Bool
    track_pairwise::Bool
end

# Default constructor
function MCMCOptions(;
    n_samples::Int = 10000,
    verbose::Bool = false,
    infer_params::Dict{Symbol, Bool} = Dict{Symbol, Bool}(
        :λ => true,
        :r => true,
        :m => true,
        :p => true,
        :c => true,
        :ξ => true,
        :ω => true,
        :α => true,
        :h => true
    ),
    prop_sds::Dict{Symbol, Float64} = Dict{Symbol, Float64}(
        :λ => 0.5,
        :r => 0.5,
        :m => 0.5,
        :p => 0.5,
        :ω => 0.3,
        :α => 0.5
    ),
    track_diagnostics::Bool = true,
    track_pairwise::Bool = false,
)
    return MCMCOptions(
        n_samples,
        verbose,
        infer_params,
        prop_sds,
        track_diagnostics,
        track_pairwise,
    )
end

"""
    should_infer(opts::MCMCOptions, param::Symbol) -> Bool

Check if a parameter should be inferred. Defaults to true if not specified.
"""
should_infer(opts::MCMCOptions, param::Symbol) = get(opts.infer_params, param, true)

"""
    get_prop_sd(opts::MCMCOptions, param::Symbol; default=0.5) -> Float64

Get the proposal standard deviation for a parameter.
"""
get_prop_sd(opts::MCMCOptions, param::Symbol; default=0.5) = get(opts.prop_sds, param, default)

