# ============================================================================
# Tests for Birth Proposals - Comprehensive Dispatch Testing
# ============================================================================

@testset "Birth Proposals" begin

    # ========================================================================
    # Moment Fitting Utilities
    # ========================================================================

    @testset "fit_inverse_gamma_moments" begin
        # Test with known data
        Random.seed!(123)
        data = rand(InverseGamma(5.0, 10.0), 1000)
        params = fit_inverse_gamma_moments(data)

        @test !isnothing(params)
        α_fit, β_fit = params
        @test α_fit > 2  # Valid InverseGamma
        @test β_fit > 0

        # Should be close to true values for large sample
        @test abs(α_fit - 5.0) < 1.0
        @test abs(β_fit - 10.0) < 2.0

        # Edge cases: too few points
        @test isnothing(fit_inverse_gamma_moments([1.0]))
        @test isnothing(fit_inverse_gamma_moments([1.0, 2.0]))

        # Edge case: zero variance
        @test isnothing(fit_inverse_gamma_moments([1.0, 1.0, 1.0]))

        # Edge case: negative mean (invalid)
        @test isnothing(fit_inverse_gamma_moments([-1.0, -2.0, -3.0, -4.0]))
    end

    @testset "fit_gamma_shape_moments" begin
        # Test with known data
        Random.seed!(123)
        data = rand(Gamma(3.0, 2.0), 1000)
        α_est = fit_gamma_shape_moments(data)

        @test !isnothing(α_est)
        @test α_est > 0

        # Should be close to true shape for large sample
        @test abs(α_est - 3.0) < 1.0

        # Edge cases
        @test isnothing(fit_gamma_shape_moments([1.0]))  # Too few points
        @test isnothing(fit_gamma_shape_moments([1.0, 1.0, 1.0]))  # Zero variance

        # Edge case: very small variance
        data_small_var = [1.0, 1.001, 0.999, 1.002]
        result = fit_gamma_shape_moments(data_small_var)
        if !isnothing(result)
            @test result > 0
        end
    end

    # ========================================================================
    # Proposal Type Constructors
    # ========================================================================

    @testset "Proposal Type Constructors" begin
        # NormalMomentMatch - single parameter
        prop1 = NormalMomentMatch(0.5)
        @test prop1.σ == [0.5]

        # NormalMomentMatch - multiple parameters
        prop2 = NormalMomentMatch(0.5, 1.0, 0.3)
        @test prop2.σ == [0.5, 1.0, 0.3]

        # InverseGammaMomentMatch - default min_size
        prop3 = InverseGammaMomentMatch()
        @test prop3.min_size == 3

        # InverseGammaMomentMatch - custom min_size
        prop4 = InverseGammaMomentMatch(10)
        @test prop4.min_size == 10

        # LogNormalMomentMatch - single parameter
        prop5 = LogNormalMomentMatch(0.5)
        @test prop5.σ == [0.5]
        @test prop5.min_size == 2

        # LogNormalMomentMatch - custom min_size
        prop6 = LogNormalMomentMatch(0.5; min_size=5)
        @test prop6.min_size == 5

        # LogNormalMomentMatch - multiple parameters
        prop7 = LogNormalMomentMatch(0.5, 0.3; min_size=3)
        @test prop7.σ == [0.5, 0.3]
        @test prop7.min_size == 3

        # FixedDistributionProposal - single distribution
        prop8 = FixedDistributionProposal(Gamma(2.0, 1.0))
        @test length(prop8.dists) == 1

        # FixedDistributionProposal - multiple distributions
        prop9 = FixedDistributionProposal([Gamma(2.0, 1.0), Normal(0.0, 1.0)])
        @test length(prop9.dists) == 2
    end

    # ========================================================================
    # Proposal Dispatch - Negative Binomial Models
    # ========================================================================

    @testset "PriorProposal - NBGammaPoissonGlobalR" begin
        Random.seed!(42)
        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1.0)
        n = 10
        c = collect(1:n)
        λ = rand(n) .+ 1.0
        m_dict = Dict{Vector{Int}, Float64}([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] => 1.5)
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 2.0)
        D = zeros(n, n)
        data = CountData(rand(1:10, n), D)
        S_i = [1, 2, 3]

        # Sample from prior
        params, log_q_fwd = sample_birth_params(model, PriorProposal(), S_i, state, data, priors)

        @test haskey(params, :m)
        @test params.m > 0
        @test isfinite(log_q_fwd)

        # Check symmetry: logpdf should match
        log_q_rev = birth_params_logpdf(model, PriorProposal(), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    @testset "NormalMomentMatch - NBGammaPoissonGlobalR" begin
        Random.seed!(42)
        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1.0)
        n = 15
        c = collect(1:n)
        λ = rand(n) .+ 2.0
        m_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] => 2.5)
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 3.0)
        D = zeros(n, n)
        data = CountData(rand(1:10, n), D)
        S_i = [1, 2, 3, 4, 5]

        # Moment-matched proposal
        params, log_q_fwd = sample_birth_params(model, NormalMomentMatch(1.0), S_i, state, data, priors)

        @test params.m > 0
        @test isfinite(log_q_fwd)

        # Check symmetry
        log_q_rev = birth_params_logpdf(model, NormalMomentMatch(1.0), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10

        # Test with small moving set (should fall back to prior mean)
        S_i_small = [1]
        params_small, log_q_small = sample_birth_params(model, NormalMomentMatch(1.0), S_i_small, state, data, priors)
        @test params_small.m > 0
        @test isfinite(log_q_small)
    end

    @testset "FixedDistributionProposal - NBGammaPoissonGlobalR" begin
        Random.seed!(42)
        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1.0)
        n = 10
        c = collect(1:n)
        λ = rand(n) .+ 1.0
        m_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 1.5)
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 2.0)
        D = zeros(n, n)
        data = CountData(rand(1:10, n), D)
        S_i = [1, 2, 3]

        # Fixed distribution proposal
        Q = Gamma(3.0, 0.5)
        params, log_q_fwd = sample_birth_params(model, FixedDistributionProposal(Q), S_i, state, data, priors)

        @test params.m > 0
        @test isfinite(log_q_fwd)

        # Check that logpdf matches distribution
        log_q_rev = birth_params_logpdf(model, FixedDistributionProposal(Q), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
        @test log_q_fwd ≈ logpdf(Q, params.m) atol=1e-10
    end

    # ========================================================================
    # Proposal Dispatch - Poisson Models
    # ========================================================================

    @testset "PriorProposal - PoissonClusterRates" begin
        Random.seed!(42)
        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        n = 10
        y = rand(1:10, n)
        D = zeros(n, n)
        data = CountData(y, D)
        c = collect(1:n)
        λ_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 5.0)
        state = PoissonClusterRatesState(c, λ_dict)
        S_i = [1, 2, 3]

        params, log_q_fwd = sample_birth_params(model, PriorProposal(), S_i, state, data, priors)

        @test haskey(params, :λ)
        @test params.λ > 0
        @test isfinite(log_q_fwd)

        # Check symmetry
        log_q_rev = birth_params_logpdf(model, PriorProposal(), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    @testset "NormalMomentMatch - PoissonClusterRates" begin
        Random.seed!(42)
        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        n = 15
        y = rand(Poisson(5), n)
        D = zeros(n, n)
        data = CountData(y, D)
        c = collect(1:n)
        λ_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] => 5.0)
        state = PoissonClusterRatesState(c, λ_dict)
        S_i = [1, 2, 3, 4, 5]

        params, log_q_fwd = sample_birth_params(model, NormalMomentMatch(1.0), S_i, state, data, priors)

        @test params.λ > 0
        @test isfinite(log_q_fwd)

        log_q_rev = birth_params_logpdf(model, NormalMomentMatch(1.0), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    # ========================================================================
    # Proposal Dispatch - Binomial Models
    # ========================================================================

    @testset "PriorProposal - BinomialClusterProb" begin
        Random.seed!(42)
        model = BinomialClusterProb()
        priors = BinomialClusterProbPriors(2.0, 2.0)
        n = 10
        N = 20
        y = rand(Binomial(N, 0.5), n)
        D = zeros(n, n)
        data = CountDataWithTrials(y, N, D)
        c = collect(1:n)
        p_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 0.5)
        state = BinomialClusterProbState(c, p_dict)
        S_i = [1, 2, 3]

        params, log_q_fwd = sample_birth_params(model, PriorProposal(), S_i, state, data, priors)

        @test haskey(params, :p)
        @test 0 < params.p < 1
        @test isfinite(log_q_fwd)

        log_q_rev = birth_params_logpdf(model, PriorProposal(), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    # ========================================================================
    # Proposal Dispatch - Continuous Models
    # ========================================================================

    @testset "PriorProposal - GammaClusterShapeMarg" begin
        Random.seed!(42)
        model = GammaClusterShapeMarg()
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)
        n = 10
        y = rand(Gamma(3.0, 1.0), n)
        D = zeros(n, n)
        data = ContinuousData(y, D)
        c = collect(1:n)
        α_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 3.0)
        state = GammaClusterShapeMargState(c, α_dict)
        S_i = [1, 2, 3]

        params, log_q_fwd = sample_birth_params(model, PriorProposal(), S_i, state, data, priors)

        @test haskey(params, :α)
        @test params.α > 0
        @test isfinite(log_q_fwd)

        log_q_rev = birth_params_logpdf(model, PriorProposal(), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    @testset "NormalMomentMatch - GammaClusterShapeMarg" begin
        Random.seed!(42)
        model = GammaClusterShapeMarg()
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)
        n = 15
        y = rand(Gamma(3.0, 1.0), n)
        D = zeros(n, n)
        data = ContinuousData(y, D)
        c = collect(1:n)
        α_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] => 3.0)
        state = GammaClusterShapeMargState(c, α_dict)
        S_i = [1, 2, 3, 4, 5]

        params, log_q_fwd = sample_birth_params(model, NormalMomentMatch(1.0), S_i, state, data, priors)

        @test params.α > 0
        @test isfinite(log_q_fwd)

        log_q_rev = birth_params_logpdf(model, NormalMomentMatch(1.0), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    @testset "LogNormalMomentMatch - GammaClusterShapeMarg" begin
        Random.seed!(42)
        model = GammaClusterShapeMarg()
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)
        n = 15
        y = rand(Gamma(3.0, 1.0), n)
        D = zeros(n, n)
        data = ContinuousData(y, D)
        c = collect(1:n)
        α_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] => 3.0)
        state = GammaClusterShapeMargState(c, α_dict)
        S_i = [1, 2, 3, 4, 5]

        params, log_q_fwd = sample_birth_params(model, LogNormalMomentMatch(0.5), S_i, state, data, priors)

        @test params.α > 0
        @test isfinite(log_q_fwd)

        log_q_rev = birth_params_logpdf(model, LogNormalMomentMatch(0.5), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    @testset "PriorProposal - SkewNormalCluster" begin
        Random.seed!(42)
        model = SkewNormalCluster()
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)
        n = 10
        y = randn(n)
        D = zeros(n, n)
        data = ContinuousData(y, D)
        c = collect(1:n)
        h = abs.(randn(n))
        ξ_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 0.0)
        ω_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 1.0)
        α_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 0.0)
        state = SkewNormalClusterState(c, h, ξ_dict, ω_dict, α_dict)
        S_i = [1, 2, 3]

        params, log_q_fwd = sample_birth_params(model, PriorProposal(), S_i, state, data, priors)

        @test haskey(params, :ξ)
        @test haskey(params, :ω)
        @test haskey(params, :α)
        @test params.ω > 0
        @test isfinite(log_q_fwd)

        log_q_rev = birth_params_logpdf(model, PriorProposal(), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    @testset "NormalMomentMatch - SkewNormalCluster" begin
        Random.seed!(42)
        model = SkewNormalCluster()
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)
        n = 15
        y = randn(n) .+ 5.0  # Shift mean to 5
        D = zeros(n, n)
        data = ContinuousData(y, D)
        c = collect(1:n)
        h = abs.(randn(n))
        ξ_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] => 5.0)
        ω_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] => 1.0)
        α_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] => 0.0)
        state = SkewNormalClusterState(c, h, ξ_dict, ω_dict, α_dict)
        S_i = [1, 2, 3, 4, 5]

        # NormalMomentMatch with 3 σ values (ξ, ω, α)
        params, log_q_fwd = sample_birth_params(model, NormalMomentMatch(1.0, 0.3, 0.5), S_i, state, data, priors)

        @test params.ω > 0
        @test isfinite(log_q_fwd)

        log_q_rev = birth_params_logpdf(model, NormalMomentMatch(1.0, 0.3, 0.5), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================

    @testset "Edge Cases - Small Moving Sets" begin
        # Test that proposals handle small moving sets gracefully
        Random.seed!(42)
        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1.0)
        n = 10
        c = collect(1:n)
        λ = rand(n) .+ 1.0
        m_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 2.0)
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 2.0)
        D = zeros(n, n)
        data = CountData(rand(1:10, n), D)

        # Single observation in moving set
        S_i_single = [1]
        params_single, log_q_single = sample_birth_params(model, NormalMomentMatch(1.0), S_i_single, state, data, priors)
        @test params_single.m > 0
        @test isfinite(log_q_single)

        # InverseGammaMomentMatch with too few points (should fall back)
        params_ig, log_q_ig = sample_birth_params(model, InverseGammaMomentMatch(5), S_i_single, state, data, priors)
        @test params_ig.m > 0
        @test isfinite(log_q_ig)
    end

    @testset "Edge Cases - Numerical Stability" begin
        # Test proposals with extreme parameter values
        Random.seed!(42)
        model = GammaClusterShapeMarg()
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        # Very small shape parameter
        n = 10
        y_small_shape = rand(Gamma(0.1, 10.0), n)
        D = zeros(n, n)
        data_small = ContinuousData(y_small_shape, D)
        c = collect(1:n)
        α_dict_small = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 0.1)
        state_small = GammaClusterShapeMargState(c, α_dict_small)
        S_i = [1, 2, 3]

        params_small, log_q_small = sample_birth_params(model, NormalMomentMatch(1.0), S_i, state_small, data_small, priors)
        @test params_small.α > 0
        @test isfinite(log_q_small)

        # Very large shape parameter
        y_large_shape = rand(Gamma(100.0, 0.1), n)
        data_large = ContinuousData(y_large_shape, D)
        α_dict_large = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 100.0)
        state_large = GammaClusterShapeMargState(c, α_dict_large)

        params_large, log_q_large = sample_birth_params(model, NormalMomentMatch(1.0), S_i, state_large, data_large, priors)
        @test params_large.α > 0
        @test isfinite(log_q_large)
    end

end
