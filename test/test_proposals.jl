# ============================================================================
# Tests for Birth Proposals - Comprehensive Dispatch Testing
# ============================================================================

@testset "Birth Proposals" begin

    # ========================================================================
    # Moment Fitting Utilities
    # ========================================================================

    @testset "fit_inverse_gamma_moments" begin
        Random.seed!(123)
        data = rand(InverseGamma(5.0, 10.0), 1000)
        params = fit_inverse_gamma_moments(data)

        @test !isnothing(params)
        α_fit, β_fit = params
        @test α_fit > 2
        @test β_fit > 0

        @test abs(α_fit - 5.0) < 1.0
        @test abs(β_fit - 10.0) < 2.0

        @test isnothing(fit_inverse_gamma_moments([1.0]))
        @test !isnothing(fit_inverse_gamma_moments([1.0, 2.0]))
        @test isnothing(fit_inverse_gamma_moments([1.0, 1.0, 1.0]))
        @test isnothing(fit_inverse_gamma_moments([-1.0, -2.0, -3.0, -4.0]))
    end

    @testset "fit_gamma_shape_moments" begin
        Random.seed!(123)
        data = rand(Gamma(3.0, 2.0), 1000)
        α_est = fit_gamma_shape_moments(data)

        @test !isnothing(α_est)
        @test α_est > 0
        @test abs(α_est - 3.0) < 1.0

        @test isnothing(fit_gamma_shape_moments([1.0]))
        @test isnothing(fit_gamma_shape_moments([1.0, 1.0, 1.0]))

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
        prop1 = NormalMomentMatch(0.5)
        @test prop1.σ == [0.5]

        prop2 = NormalMomentMatch(0.5, 1.0, 0.3)
        @test prop2.σ == [0.5, 1.0, 0.3]

        prop3 = InverseGammaMomentMatch()
        @test prop3.min_size == 3

        prop4 = InverseGammaMomentMatch(10)
        @test prop4.min_size == 10

        prop5 = LogNormalMomentMatch(0.5)
        @test prop5.σ == [0.5]
        @test prop5.min_size == 2

        prop6 = LogNormalMomentMatch(0.5; min_size=5)
        @test prop6.min_size == 5

        prop7 = LogNormalMomentMatch([0.5, 0.3]; min_size=3)
        @test prop7.σ == [0.5, 0.3]
        @test prop7.min_size == 3

        prop8 = FixedDistributionProposal(Gamma(2.0, 1.0))
        @test length(prop8.dists) == 1

        prop9 = FixedDistributionProposal([Gamma(2.0, 1.0), Normal(0.0, 1.0)])
        @test length(prop9.dists) == 2
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

        log_q_rev = birth_params_logpdf(model, PriorProposal(), params, S_i, state, data, priors)
        @test log_q_fwd ≈ log_q_rev atol=1e-10
    end

    @testset "NormalMomentMatch - PoissonClusterRates" begin
        @test_broken false  # Not implemented for PoissonClusterRates
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

    # ========================================================================
    # MixedProposal
    # ========================================================================

    @testset "MixedProposal - Constructor" begin
        prop = MixedProposal(ξ = PriorProposal(), ω = InverseGammaMomentMatch(3), α = NormalMomentMatch(0.5))
        @test prop isa MixedProposal
        @test prop.proposals.ξ isa PriorProposal
        @test prop.proposals.ω isa InverseGammaMomentMatch
        @test prop.proposals.α isa NormalMomentMatch

        prop2 = MixedProposal((ξ = PriorProposal(), ω = PriorProposal(), α = PriorProposal()))
        @test prop2 isa MixedProposal
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================

    @testset "Edge Cases - Numerical Stability" begin
        Random.seed!(42)
        model = GammaClusterShapeMarg()
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

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

        y_large_shape = rand(Gamma(100.0, 0.1), n)
        data_large = ContinuousData(y_large_shape, D)
        α_dict_large = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 100.0)
        state_large = GammaClusterShapeMargState(c, α_dict_large)

        params_large, log_q_large = sample_birth_params(model, NormalMomentMatch(1.0), S_i, state_large, data_large, priors)
        @test params_large.α > 0
        @test isfinite(log_q_large)
    end

end
