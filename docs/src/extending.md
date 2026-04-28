```@meta
CurrentModule = DistanceDependentCRP
```

# Adding Your Own Model

New likelihood families can be added by implementing a small set of interface methods. The DDCRP inference machinery (Gibbs, RJMCMC, hyperparameter samplers, diagnostics) is generic and will work automatically once these methods are defined.

## Step 1 — Define the Four Core Structs

Every model needs four concrete types, each subtyping the appropriate abstract base.

```julia
# 1. Model tag — identifies the likelihood family
struct MyModel <: LikelihoodModel end

# 2. MCMC state — c is always required; add parameter dicts for RJMCMC
mutable struct MyModelState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}                          # customer → customer links
    θ_dict::Dict{Vector{Int}, T}            # cluster key → parameter (RJMCMC only)
end

# 3. Priors
struct MyModelPriors{T<:Real} <: AbstractPriors
    a::T
    b::T
end

# 4. Sample storage
struct MyModelSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}        # n_samples × n_obs
    θ::Matrix{T}          # n_samples × n_obs  (omit if marginalised)
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
end
```

**Cluster keys**: cluster parameters are stored in `Dict{Vector{Int}, T}` where the key is the **sorted vector of customer indices** belonging to that cluster. Use `table_vector(c)` to obtain these keys from the assignment vector `c`.

## Step 2 — Implement Required Interface Methods

### Always Required

**`initialise_state`** — Construct the initial state from data and priors.

```julia
function initialise_state(
    model::MyModel,
    data::AbstractObservedData,
    ddcrp_params::DDCRPParams,
    priors::MyModelPriors
)
    n = nobs(data)
    c = collect(1:n)          # each customer links to themselves initially
    # initialise parameter dict from single-observation clusters
    tables = table_vector(c)
    θ_dict = Dict(t => rand(Gamma(priors.a, 1/priors.b)) for t in tables)
    return MyModelState(c, θ_dict)
end
```

**`allocate_samples`** — Pre-allocate the samples container.

```julia
function allocate_samples(model::MyModel, n_samples::Int, n_obs::Int)
    return MyModelSamples(
        zeros(Int, n_samples, n_obs),
        zeros(Float64, n_samples, n_obs),
        zeros(Float64, n_samples),
        zeros(Float64, n_samples),
        zeros(Float64, n_samples),
    )
end
```

**`extract_samples!`** — Copy the current state into the samples arrays at iteration `iter`.

```julia
function extract_samples!(
    model::MyModel,
    state::MyModelState,
    samples::MyModelSamples,
    iter::Int
)
    samples.c[iter, :] = state.c
    for (table, θ) in state.θ_dict
        samples.θ[iter, table] .= θ
    end
    # logpost, α_ddcrp, s_ddcrp are filled by the main MCMC loop
end
```

**`update_params!`** — Update latent cluster parameters within one MCMC iteration.

```julia
function update_params!(
    model::MyModel,
    state::MyModelState,
    data::AbstractObservedData,
    priors::MyModelPriors,
    tables::Vector{Vector{Int}},
    log_DDCRP::AbstractMatrix,
    opts::MCMCOptions
)
    y = observations(data)
    for table in tables
        # Conjugate posterior update example
        a_post = priors.a + sum(y[table])
        b_post = priors.b + length(table)
        state.θ_dict[table] = rand(Gamma(a_post, 1/b_post))
    end
end
```

**`table_contribution`** — Log-likelihood contribution of a single cluster.

For a **marginalised** model return the marginal log-likelihood (integrate out the cluster parameters). For a **non-marginalised** model return the conditional log-likelihood given the current parameter value.

```julia
# Marginalised
function table_contribution(
    model::MyModel,
    table::AbstractVector{Int},
    state::MyModelState,
    data::AbstractObservedData,
    priors::MyModelPriors
)
    y = observations(data)
    n_k = length(table)
    s_k = sum(y[table])
    # Gamma–Poisson marginal log-likelihood
    return (
        logabsgamma(priors.a + s_k)[1] - logabsgamma(priors.a)[1]
        + priors.a * log(priors.b)
        - (priors.a + s_k) * log(priors.b + n_k)
        - sum(logfactorial.(y[table]))
    )
end
```

**`posterior`** — Full log-posterior for diagnostics.

```julia
function posterior(
    model::MyModel,
    data::AbstractObservedData,
    state::MyModelState,
    priors::MyModelPriors,
    log_DDCRP::AbstractMatrix
)
    tables = table_vector(state.c)
    lp = ddcrp_contribution(state.c, log_DDCRP)
    for table in tables
        lp += table_contribution(model, table, state, data, priors)
    end
    return lp
end
```

---

### RJMCMC Only

The following three methods are required **only** for non-marginalised models that use RJMCMC (i.e. any `BirthProposal` other than `ConjugateProposal`).

**`cluster_param_dicts`** — Return a `NamedTuple` of parameter dictionaries, one entry per cluster parameter.

```julia
function cluster_param_dicts(state::MyModelState)
    return (θ = state.θ_dict,)
end
```

**`sample_birth_params`** — Sample new cluster parameters for a birth move, given the moving set `S_i`. Return `(params_nt, log_q)` where `params_nt` is a `NamedTuple` of sampled values and `log_q` is the log proposal density.

```julia
function sample_birth_params(
    model::MyModel,
    proposal::BirthProposal,
    S_i::Vector{Int},
    state::MyModelState,
    data::AbstractObservedData,
    priors::MyModelPriors
)
    θ_new, log_q = sample_birth_param(model, Val(:θ), proposal, S_i, state, data, priors)
    return (θ = θ_new,), log_q
end
```

**`birth_params_logpdf`** — Log-density of the birth proposal at given parameter values. Used when computing the reverse (death) proposal probability.

```julia
function birth_params_logpdf(
    model::MyModel,
    proposal::BirthProposal,
    params::Vector,
    S_i::Vector{Int},
    state::MyModelState,
    data::AbstractObservedData,
    priors::MyModelPriors
)
    return birth_param_logpdf(model, Val(:θ), proposal, params[1], S_i, state, data, priors)
end
```

## Step 3 — Choose Inference Mode

| Scenario | Birth proposal | Extra methods needed |
|---|---|---|
| Marginalised (Gibbs) | `ConjugateProposal()` | None |
| Non-marginalised (RJMCMC) | Anything else | `cluster_param_dicts`, `sample_birth_params`, `birth_params_logpdf` |

Pass the chosen proposal as the fifth argument to `mcmc`:

```julia
# Gibbs (marginalised)
samples = mcmc(MyModel(), data, ddcrp, priors, ConjugateProposal(); opts=opts)

# RJMCMC (non-marginalised)
samples, diag = mcmc(
    MyModel(), data, ddcrp, priors, LogNormalMomentMatch(0.5);
    fixed_dim_proposal=WeightedMean(), opts=opts
)
```

## Complete Skeleton

```julia
using DistanceDependentCRP
using Distributions, SpecialFunctions

# ── Types ──────────────────────────────────────────────────────────────────────

struct MyModel <: LikelihoodModel end

mutable struct MyModelState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    θ_dict::Dict{Vector{Int}, T}
end

struct MyModelPriors{T<:Real} <: AbstractPriors
    a::T
    b::T
end

struct MyModelSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    θ::Matrix{T}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
end

# ── Interface ──────────────────────────────────────────────────────────────────

function DDCRP.initialise_state(model::MyModel, data, ddcrp_params, priors::MyModelPriors)
    n      = nobs(data)
    c      = collect(1:n)
    tables = table_vector(c)
    θ_dict = Dict(t => rand(Gamma(priors.a, 1/priors.b)) for t in tables)
    return MyModelState(c, θ_dict)
end

function DDCRP.allocate_samples(model::MyModel, n_samples::Int, n_obs::Int)
    T = Float64
    return MyModelSamples(
        zeros(Int, n_samples, n_obs),
        zeros(T, n_samples, n_obs),
        zeros(T, n_samples),
        zeros(T, n_samples),
        zeros(T, n_samples),
    )
end

function DDCRP.extract_samples!(model::MyModel, state::MyModelState, samples::MyModelSamples, iter::Int)
    samples.c[iter, :] = state.c
    for (table, θ) in state.θ_dict
        samples.θ[iter, table] .= θ
    end
end

function DDCRP.update_params!(model::MyModel, state::MyModelState, data, priors::MyModelPriors, tables, log_DDCRP, opts)
    y = observations(data)
    for table in tables
        a_post = priors.a + sum(y[table])
        b_post = priors.b + length(table)
        state.θ_dict[table] = rand(Gamma(a_post, 1/b_post))
    end
end

function DDCRP.table_contribution(model::MyModel, table, state::MyModelState, data, priors::MyModelPriors)
    # Replace with your marginal or conditional log-likelihood
    y   = observations(data)
    θ   = state.θ_dict[sort(collect(table))]
    return sum(logpdf.(Poisson(θ), y[table]))
end

function DDCRP.posterior(model::MyModel, data, state::MyModelState, priors::MyModelPriors, log_DDCRP)
    tables = table_vector(state.c)
    lp     = ddcrp_contribution(state.c, log_DDCRP)
    for t in tables
        lp += table_contribution(model, t, state, data, priors)
    end
    return lp
end

# RJMCMC interface (omit if using ConjugateProposal)

function DDCRP.cluster_param_dicts(state::MyModelState)
    return (θ = state.θ_dict,)
end
```
