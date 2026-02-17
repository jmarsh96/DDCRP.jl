# ============================================================================
# Test Helper Functions
# ============================================================================
#
# Shared utilities for testing DDCRP.jl functionality

using DDCRP
using Statistics
using Distributions

# ============================================================================
# State Validation
# ============================================================================

"""
    check_state_consistency(model, state, data)

Validate that a state object is internally consistent.
Returns nothing if valid, throws an error otherwise.
"""
function check_state_consistency(model::DDCRP.LikelihoodModel, state::DDCRP.AbstractMCMCState, data::DDCRP.AbstractObservedData)
    n = length(state.c)
    dicts = DDCRP.cluster_param_dicts(state)

    # Check all observations are covered
    tables = DDCRP.table_vector(state.c)
    all_obs = sort(vcat(tables...))
    @assert all_obs == 1:n "All observations must be in some table"

    # Check dict keys are sorted and cover all observations
    for (param_name, param_dict) in pairs(dicts)
        @assert !isempty(param_dict) "Parameter dict $param_name should not be empty"

        # Check keys are sorted
        for table in keys(param_dict)
            @assert issorted(table) "Table keys must be sorted in $param_name"
        end

        # Check all observations covered
        obs_in_dicts = sort(vcat(collect(keys(param_dict))...))
        @assert obs_in_dicts == 1:n "All observations must be in $param_name dict"
    end

    # Model-specific parameter constraints
    check_parameter_constraints(model, state)
end

"""
    check_parameter_constraints(model, state)

Check model-specific parameter constraints (dispatch on model type).
"""
function check_parameter_constraints(::DDCRP.LikelihoodModel, ::DDCRP.AbstractMCMCState)
    # Default: no specific constraints
    return true
end

# Negative Binomial: m > 0, r > 0
function check_parameter_constraints(::DDCRP.NegativeBinomialModel, state::DDCRP.AbstractMCMCState)
    if hasproperty(state, :λ)
        @assert all(state.λ .> 0) "Latent rates λ must be positive"
    end
    if hasproperty(state, :r)
        @assert state.r > 0 "Dispersion r must be positive"
    end
    if hasproperty(state, :m_dict)
        for m in values(state.m_dict)
            @assert m > 0 "Cluster means must be positive"
        end
    end
end

# Poisson: λ > 0
function check_parameter_constraints(::DDCRP.PoissonModel, state::DDCRP.AbstractMCMCState)
    if hasproperty(state, :λ_dict)
        for λ in values(state.λ_dict)
            @assert λ > 0 "Cluster rates must be positive"
        end
    end
    if hasproperty(state, :ρ_dict)
        for ρ in values(state.ρ_dict)
            @assert ρ > 0 "Population rates must be positive"
        end
    end
end

# Binomial: 0 < p < 1
function check_parameter_constraints(::DDCRP.BinomialModel, state::DDCRP.AbstractMCMCState)
    if hasproperty(state, :p_dict)
        for p in values(state.p_dict)
            @assert 0 < p < 1 "Cluster probabilities must be in (0, 1)"
        end
    end
end

# Gamma: α > 0
function check_parameter_constraints(::DDCRP.GammaModel, state::DDCRP.AbstractMCMCState)
    if hasproperty(state, :α_dict)
        for α in values(state.α_dict)
            @assert α > 0 "Gamma shape parameters must be positive"
        end
    end
end

# Skew Normal: ω > 0
function check_parameter_constraints(::DDCRP.SkewNormalModel, state::DDCRP.AbstractMCMCState)
    if hasproperty(state, :ω_dict)
        for ω in values(state.ω_dict)
            @assert ω > 0 "Scale parameters must be positive"
        end
    end
end

# ============================================================================
# Posterior Validation
# ============================================================================

"""
    check_posterior_components(model, state, data, priors, log_DDCRP)

Verify that posterior calculation is correct by checking components.
Returns nothing if valid, throws an error otherwise.
"""
function check_posterior_components(model::DDCRP.LikelihoodModel, state::DDCRP.AbstractMCMCState,
                                     data::DDCRP.AbstractObservedData, priors::DDCRP.AbstractPriors,
                                     log_DDCRP::AbstractMatrix)
    # Compute full posterior
    full_post = DDCRP.posterior(model, data, state, priors, log_DDCRP)
    @assert isfinite(full_post) "Posterior must be finite"

    # Compute components separately
    dicts = DDCRP.cluster_param_dicts(state)
    primary_dict = first(dicts)

    table_sum = sum(DDCRP.table_contribution(model, table, state, data, priors)
                    for table in keys(primary_dict))
    @assert isfinite(table_sum) "Sum of table contributions must be finite"

    ddcrp_contrib = DDCRP.ddcrp_contribution(state.c, log_DDCRP)
    @assert isfinite(ddcrp_contrib) "DDCRP contribution must be finite"

    # Check they match
    expected_post = table_sum + ddcrp_contrib
    @assert abs(full_post - expected_post) < 1e-10 "Posterior should equal sum of components"

    return true
end

# ============================================================================
# Parameter Recovery Testing
# ============================================================================

"""
    test_parameter_recovery(param_samples, true_param; tol=0.3, param_name="parameter")

Test that posterior samples recover a known true parameter.
Returns a NamedTuple with recovery statistics.
"""
function test_parameter_recovery(param_samples::AbstractVector, true_param::Real;
                                  tol::Real=0.3, param_name::String="parameter")
    # Remove burnin (first 20%)
    n_samples = length(param_samples)
    burnin = div(n_samples, 5)
    samples_post_burnin = param_samples[(burnin+1):end]

    # Compute posterior mean
    post_mean = mean(samples_post_burnin)
    rel_error = abs(post_mean - true_param) / abs(true_param)

    # Check 95% credible interval
    q025 = quantile(samples_post_burnin, 0.025)
    q975 = quantile(samples_post_burnin, 0.975)
    contains_truth = q025 <= true_param <= q975

    return (mean=post_mean, ci_lower=q025, ci_upper=q975, rel_error=rel_error,
            within_tolerance=(rel_error < tol), truth_in_ci=contains_truth)
end

"""
    test_cluster_recovery(c_samples, c_true; min_ari=0.5)

Test that posterior clustering recovers true clustering.
Returns a NamedTuple with ARI statistics.
"""
function test_cluster_recovery(c_samples::AbstractMatrix{Int}, c_true::AbstractVector{Int};
                                min_ari::Real=0.5)
    # Compute ARI trace
    ari_trace = DDCRP.compute_ari_trace(c_samples, c_true)

    # Remove burnin
    n_samples = size(c_samples, 1)
    burnin = div(n_samples, 5)
    ari_post_burnin = ari_trace[(burnin+1):end]

    # Compute mean ARI
    mean_ari = mean(ari_post_burnin)

    return (mean_ari=mean_ari, max_ari=maximum(ari_post_burnin),
            meets_threshold=(mean_ari >= min_ari))
end

# ============================================================================
# MCMC Diagnostics Helpers
# ============================================================================

"""
    check_acceptance_rates(diag; min_rate=0.05, max_rate=0.70)

Validate that MCMC acceptance rates are in a reasonable range.
Returns the acceptance rates NamedTuple.
"""
function check_acceptance_rates(diag::DDCRP.MCMCDiagnostics;
                                  min_rate::Real=0.05, max_rate::Real=0.70)
    rates = DDCRP.acceptance_rates(diag)

    # Overall rate should be in reasonable range (unless no moves)
    if !isnan(rates.overall)
        @assert min_rate <= rates.overall <= max_rate "Overall acceptance rate $(rates.overall) not in [$min_rate, $max_rate]"
    end

    # Individual move types (if they occurred)
    for (move_type, rate) in pairs(rates)
        if move_type != :overall && !isnan(rate)
            @assert 0 <= rate <= 1 "$move_type acceptance rate $rate not in [0, 1]"
        end
    end

    return rates
end

"""
    check_mixing(samples; max_lag=50, target_acf=0.3)

Check that MCMC chain is mixing reasonably well.
Returns the autocorrelation function.
"""
function check_mixing(samples::AbstractVector; max_lag::Int=50, target_acf::Real=0.3)
    # Compute autocorrelation
    acf = DDCRP.autocorrelation(samples, max_lag)

    # Check that autocorrelation is approximately 1 at lag 0
    @assert abs(acf[1] - 1.0) < 1e-6 "ACF at lag 0 should be 1"

    # Check that autocorrelation is below target at max_lag (if we have enough lags)
    if length(acf) >= max_lag
        @assert abs(acf[max_lag]) < target_acf "ACF at lag $max_lag is $(acf[max_lag]), should be < $target_acf"
    end

    return acf
end

# ============================================================================
# Numerical Stability Helpers
# ============================================================================

"""
    check_numerical_stability(values; name="values")

Check that numerical values are finite and not NaN.
Returns nothing if valid, throws an error otherwise.
"""
function check_numerical_stability(values::Union{Real, AbstractArray}; name::String="values")
    if values isa Real
        @assert isfinite(values) "$name should be finite, got $values"
    else
        @assert all(isfinite.(values)) "All $name should be finite"
        @assert !any(isnan.(values)) "No $name should be NaN"
    end
    return true
end

# ============================================================================
# Edge Case Data Generators
# ============================================================================

"""
    create_single_cluster_data(model_type, n)

Create data where all observations belong to one cluster.
"""
function create_single_cluster_data(::Type{<:DDCRP.NegativeBinomialModel}, n::Int)
    c = collect(1:n)
    c[2:end] .= 1  # All point to customer 1
    D = zeros(n, n)
    y = rand(Poisson(5), n)
    return (c=c, D=D, y=y)
end

function create_single_cluster_data(::Type{<:DDCRP.BinomialModel}, n::Int)
    c = collect(1:n)
    c[2:end] .= 1
    D = zeros(n, n)
    N = 10
    y = rand(Binomial(N, 0.5), n)
    return (c=c, D=D, y=y, N=N)
end

function create_single_cluster_data(::Type{<:DDCRP.GammaModel}, n::Int)
    c = collect(1:n)
    c[2:end] .= 1
    D = zeros(n, n)
    y = rand(Gamma(2.0, 1.0), n)
    return (c=c, D=D, y=y)
end

"""
    create_all_singletons_data(model_type, n)

Create data where each observation is its own cluster.
"""
function create_all_singletons_data(::Type{<:DDCRP.NegativeBinomialModel}, n::Int)
    c = collect(1:n)  # All self-loops
    D = ones(n, n) * 100  # Large distances
    for i in 1:n
        D[i, i] = 0.0
    end
    y = rand(Poisson(5), n)
    return (c=c, D=D, y=y)
end

function create_all_singletons_data(::Type{<:DDCRP.GammaModel}, n::Int)
    c = collect(1:n)
    D = ones(n, n) * 100
    for i in 1:n
        D[i, i] = 0.0
    end
    y = rand(Gamma(2.0, 1.0), n)
    return (c=c, D=D, y=y)
end
