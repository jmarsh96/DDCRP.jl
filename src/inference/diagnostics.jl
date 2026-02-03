# ============================================================================
# MCMC Diagnostics
# Acceptance rates, ESS, IAT calculations
# ============================================================================

using Statistics

"""
    MCMCDiagnostics

Container for MCMC diagnostic information including move-type acceptance
rates and optional pairwise proposal/acceptance tracking.
"""
mutable struct MCMCDiagnostics
    # Move-type counts
    birth_proposes::Int
    birth_accepts::Int
    death_proposes::Int
    death_accepts::Int
    fixed_proposes::Int
    fixed_accepts::Int

    # Pairwise tracking (optional, can be expensive for large n)
    propose_matrix::Union{Matrix{Int}, Nothing}
    accept_matrix::Union{Matrix{Int}, Nothing}

    # Timing
    start_time::Float64
    total_time::Float64
end

function MCMCDiagnostics(n::Int; track_pairwise::Bool=false)
    pm = track_pairwise ? zeros(Int, n, n) : nothing
    am = track_pairwise ? zeros(Int, n, n) : nothing
    MCMCDiagnostics(0, 0, 0, 0, 0, 0, pm, am, time(), 0.0)
end

"""
    record_move!(diag::MCMCDiagnostics, move_type::Symbol, accepted::Bool)

Record a proposed move and whether it was accepted.
"""
function record_move!(diag::MCMCDiagnostics, move_type::Symbol, accepted::Bool)
    if move_type == :birth
        diag.birth_proposes += 1
        accepted && (diag.birth_accepts += 1)
    elseif move_type == :death
        diag.death_proposes += 1
        accepted && (diag.death_accepts += 1)
    elseif move_type == :fixed
        diag.fixed_proposes += 1
        accepted && (diag.fixed_accepts += 1)
    end
end

"""
    record_pairwise!(diag::MCMCDiagnostics, i::Int, j::Int, accepted::Bool)

Record a pairwise proposal (i proposed to link to j).
"""
function record_pairwise!(diag::MCMCDiagnostics, i::Int, j::Int, accepted::Bool)
    if !isnothing(diag.propose_matrix)
        diag.propose_matrix[i, j] += 1
        accepted && (diag.accept_matrix[i, j] += 1)
    end
end

"""
    finalize!(diag::MCMCDiagnostics)

Finalize diagnostics by recording total time.
"""
function finalize!(diag::MCMCDiagnostics)
    diag.total_time = time() - diag.start_time
end

# ============================================================================
# Acceptance Rates
# ============================================================================

"""
    acceptance_rates(diag::MCMCDiagnostics)

Compute acceptance rates for each move type and overall.
"""
function acceptance_rates(diag::MCMCDiagnostics)
    birth_rate = diag.birth_proposes > 0 ? diag.birth_accepts / diag.birth_proposes : NaN
    death_rate = diag.death_proposes > 0 ? diag.death_accepts / diag.death_proposes : NaN
    fixed_rate = diag.fixed_proposes > 0 ? diag.fixed_accepts / diag.fixed_proposes : NaN

    total_proposes = diag.birth_proposes + diag.death_proposes + diag.fixed_proposes
    total_accepts = diag.birth_accepts + diag.death_accepts + diag.fixed_accepts
    overall = total_proposes > 0 ? total_accepts / total_proposes : NaN

    return (
        birth = birth_rate,
        death = death_rate,
        fixed = fixed_rate,
        overall = overall
    )
end

"""
    pairwise_acceptance_rates(diag::MCMCDiagnostics)

Compute pairwise acceptance rate matrix.
"""
function pairwise_acceptance_rates(diag::MCMCDiagnostics)
    if isnothing(diag.propose_matrix)
        error("Pairwise tracking was not enabled")
    end

    rates = similar(diag.propose_matrix, Float64)
    for i in axes(rates, 1), j in axes(rates, 2)
        if diag.propose_matrix[i, j] > 0
            rates[i, j] = diag.accept_matrix[i, j] / diag.propose_matrix[i, j]
        else
            rates[i, j] = NaN
        end
    end
    return rates
end

# ============================================================================
# Autocorrelation and Effective Sample Size
# ============================================================================

"""
    autocorrelation(x::AbstractVector, max_lag::Int)

Compute autocorrelation function for lags 0 to max_lag.
"""
function autocorrelation(x::AbstractVector, max_lag::Int)
    n = length(x)
    x_centered = x .- mean(x)
    var_x = var(x)

    if var_x ≈ 0
        return ones(max_lag + 1)  # Constant series
    end

    acf = zeros(max_lag + 1)
    acf[1] = 1.0  # lag 0

    for k in 1:max_lag
        acf[k + 1] = sum(x_centered[1:n-k] .* x_centered[k+1:n]) / ((n - k) * var_x)
    end

    return acf
end

"""
    integrated_autocorrelation_time(x::AbstractVector; max_lag=nothing, method=:initial_positive)

Compute Integrated Autocorrelation Time (IAT).
τ = 1 + 2 * Σ_{k=1}^{K} ρ(k)

# Methods
- `:simple` - Sum until first negative autocorrelation
- `:initial_positive` - Geyer's initial positive sequence estimator (recommended)
- `:batch` - Batch means estimator
"""
function integrated_autocorrelation_time(x::AbstractVector; max_lag=nothing, method=:initial_positive)
    n = length(x)

    if isnothing(max_lag)
        max_lag = min(n ÷ 2, 1000)
    end

    acf = autocorrelation(x, max_lag)

    if method == :simple
        τ = 1.0
        for k in 1:max_lag
            if acf[k + 1] < 0
                break
            end
            τ += 2 * acf[k + 1]
        end
        return τ

    elseif method == :initial_positive
        # Geyer's initial positive sequence estimator
        τ = acf[1]  # = 1.0
        for k in 1:2:max_lag-1
            gamma_k = acf[k + 1] + acf[k + 2]
            if gamma_k < 0
                break
            end
            τ += 2 * gamma_k
        end
        return τ

    elseif method == :batch
        batch_size = Int(ceil(sqrt(n)))
        n_batches = n ÷ batch_size
        if n_batches < 2
            return 1.0
        end
        batch_means = [mean(x[(i-1)*batch_size + 1 : i*batch_size]) for i in 1:n_batches]

        var_batch = var(batch_means)
        var_x = var(x)

        if var_x ≈ 0
            return 1.0
        end

        return batch_size * var_batch / var_x
    else
        error("Unknown method: $method")
    end
end

"""
    effective_sample_size(x::AbstractVector; kwargs...)

Compute Effective Sample Size: ESS = N / τ
where τ is the integrated autocorrelation time.
"""
function effective_sample_size(x::AbstractVector; kwargs...)
    n = length(x)
    τ = integrated_autocorrelation_time(x; kwargs...)
    return n / τ
end

"""
    ess_per_second(x::AbstractVector, total_time::Float64; kwargs...)

Compute ESS per second of computation time.
"""
function ess_per_second(x::AbstractVector, total_time::Float64; kwargs...)
    ess = effective_sample_size(x; kwargs...)
    return ess / total_time
end

# ============================================================================
# Summary Diagnostics
# ============================================================================

"""
    MCMCSummary

Summary statistics for an MCMC run.
"""
struct MCMCSummary
    acc_rates::NamedTuple
    ess_n_clusters::Float64
    ess_logpost::Float64
    ess_r::Float64
    iat_n_clusters::Float64
    iat_logpost::Float64
    iat_r::Float64
    total_time::Float64
    ess_per_sec_n_clusters::Float64
    total_proposals::Int
    birth_fraction::Float64
    death_fraction::Float64
    fixed_fraction::Float64
end

"""
    summarize_mcmc(samples::MCMCSamples, diag::MCMCDiagnostics)

Compute comprehensive summary of MCMC run.
"""
function summarize_mcmc(samples::AbstractMCMCSamples, diag::MCMCDiagnostics)
    n_clusters = calculate_n_clusters(samples.c)

    acc_rates = acceptance_rates(diag)

    ess_nc = effective_sample_size(Float64.(n_clusters))
    ess_lp = effective_sample_size(samples.logpost)
    ess_r_val = !isnothing(samples.r) ? effective_sample_size(samples.r) : NaN

    iat_nc = integrated_autocorrelation_time(Float64.(n_clusters))
    iat_lp = integrated_autocorrelation_time(samples.logpost)
    iat_r_val = !isnothing(samples.r) ? integrated_autocorrelation_time(samples.r) : NaN

    total_props = diag.birth_proposes + diag.death_proposes + diag.fixed_proposes

    MCMCSummary(
        acc_rates,
        ess_nc, ess_lp, ess_r_val,
        iat_nc, iat_lp, iat_r_val,
        diag.total_time,
        diag.total_time > 0 ? ess_nc / diag.total_time : 0.0,
        total_props,
        total_props > 0 ? diag.birth_proposes / total_props : 0.0,
        total_props > 0 ? diag.death_proposes / total_props : 0.0,
        total_props > 0 ? diag.fixed_proposes / total_props : 0.0
    )
end

function Base.show(io::IO, s::MCMCSummary)
    println(io, "MCMC Summary")
    println(io, "============")
    println(io, "Acceptance Rates:")
    println(io, "  Birth:   $(round(s.acc_rates.birth * 100, digits=1))%")
    println(io, "  Death:   $(round(s.acc_rates.death * 100, digits=1))%")
    println(io, "  Fixed:   $(round(s.acc_rates.fixed * 100, digits=1))%")
    println(io, "  Overall: $(round(s.acc_rates.overall * 100, digits=1))%")
    println(io, "")
    println(io, "Move Type Distribution:")
    println(io, "  Birth: $(round(s.birth_fraction * 100, digits=1))%")
    println(io, "  Death: $(round(s.death_fraction * 100, digits=1))%")
    println(io, "  Fixed: $(round(s.fixed_fraction * 100, digits=1))%")
    println(io, "")
    println(io, "Effective Sample Size:")
    println(io, "  N clusters: $(round(s.ess_n_clusters, digits=1))")
    println(io, "  Log-post:   $(round(s.ess_logpost, digits=1))")
    println(io, "  r:          $(round(s.ess_r, digits=1))")
    println(io, "")
    println(io, "Integrated Autocorrelation Time:")
    println(io, "  N clusters: $(round(s.iat_n_clusters, digits=1))")
    println(io, "  Log-post:   $(round(s.iat_logpost, digits=1))")
    println(io, "  r:          $(round(s.iat_r, digits=1))")
    println(io, "")
    println(io, "Timing:")
    println(io, "  Total time: $(round(s.total_time, digits=1))s")
    println(io, "  ESS/s (K):  $(round(s.ess_per_sec_n_clusters, digits=2))")
end
