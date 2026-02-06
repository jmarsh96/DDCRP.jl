# ============================================================================
# Tests for Birth Proposals (new dispatch-based interface)
# ============================================================================

@testset "Birth Proposals" begin

    @testset "fit_inverse_gamma_moments" begin
        # Test with known data
        data = rand(InverseGamma(5.0, 10.0), 1000)
        params = fit_inverse_gamma_moments(data)

        @test !isnothing(params)
        α_fit, β_fit = params
        @test α_fit > 2  # Valid InverseGamma
        @test β_fit > 0

        # Should be close to true values for large sample
        @test abs(α_fit - 5.0) < 1.0
        @test abs(β_fit - 10.0) < 2.0

        # Test failure cases
        @test isnothing(fit_inverse_gamma_moments([1.0]))  # Too few points
        @test isnothing(fit_inverse_gamma_moments([1.0, 1.0, 1.0]))  # Zero variance
    end

    @testset "fit_gamma_shape_moments" begin
        # Test with known data
        data = rand(Gamma(3.0, 2.0), 1000)
        α_est = fit_gamma_shape_moments(data)

        @test !isnothing(α_est)
        @test α_est > 0

        # Should be close to true shape for large sample
        @test abs(α_est - 3.0) < 1.0

        # Test failure cases
        @test isnothing(fit_gamma_shape_moments([1.0]))  # Too few points
        @test isnothing(fit_gamma_shape_moments([1.0, 1.0, 1.0]))  # Zero variance
    end

    @testset "Proposal Type Constructors" begin
        # NormalMomentMatch
        prop1 = NormalMomentMatch(0.5)
        @test prop1.σ == [0.5]

        prop2 = NormalMomentMatch(0.5, 1.0)
        @test prop2.σ == [0.5, 1.0]

        # InverseGammaMomentMatch
        prop3 = InverseGammaMomentMatch()
        @test prop3.min_size == 3

        prop4 = InverseGammaMomentMatch(5)
        @test prop4.min_size == 5

        # LogNormalMomentMatch
        prop5 = LogNormalMomentMatch(0.5)
        @test prop5.σ == [0.5]
        @test prop5.min_size == 2

        prop6 = LogNormalMomentMatch(0.5; min_size=5)
        @test prop6.min_size == 5

        # FixedDistributionProposal
        prop7 = FixedDistributionProposal(Gamma(2.0, 1.0))
        @test length(prop7.dists) == 1

        prop8 = FixedDistributionProposal([Gamma(2.0, 1.0), Normal(0.0, 1.0)])
        @test length(prop8.dists) == 2
    end

    @testset "sample_birth_params - NBGammaPoissonGlobalR" begin
        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1.0)
        n = 10
        c = collect(1:n)
        λ = rand(n) .+ 1.0
        m_dict = Dict{Vector{Int}, Float64}(sort(collect(1:n)) => 1.0)
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)
        D = zeros(n, n)
        data = CountData(rand(1:10, n), D)
        S_i = [1, 2, 3]

        # PriorProposal
        Random.seed!(42)
        params, log_q = sample_birth_params(model, PriorProposal(), S_i, state, data, priors)
        @test haskey(params, :m)
        @test params.m > 0
        @test isfinite(log_q)

        # birth_params_logpdf should match
        log_q_check = birth_params_logpdf(model, PriorProposal(), params, S_i, state, data, priors)
        @test log_q ≈ log_q_check

        # NormalMomentMatch
        params2, log_q2 = sample_birth_params(model, NormalMomentMatch(1.0), S_i, state, data, priors)
        @test params2.m > 0
        @test isfinite(log_q2)
        log_q2_check = birth_params_logpdf(model, NormalMomentMatch(1.0), params2, S_i, state, data, priors)
        @test log_q2 ≈ log_q2_check

        # FixedDistributionProposal
        params3, log_q3 = sample_birth_params(model, FixedDistributionProposal(Gamma(2.0, 1.0)), S_i, state, data, priors)
        @test params3.m > 0
        @test isfinite(log_q3)
    end

    @testset "sample_birth_params - PoissonClusterRates" begin
        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        n = 10
        y = rand(1:10, n)
        D = zeros(n, n)
        data = CountData(y, D)
        c = collect(1:n)
        λ_dict = Dict{Vector{Int}, Float64}(sort(collect(1:n)) => 2.0)
        state = PoissonClusterRatesState(c, λ_dict)
        S_i = [1, 2, 3]

        # PriorProposal (conjugate posterior)
        Random.seed!(42)
        params, log_q = sample_birth_params(model, PriorProposal(), S_i, state, data, priors)
        @test haskey(params, :λ)
        @test params.λ > 0
        @test isfinite(log_q)

        log_q_check = birth_params_logpdf(model, PriorProposal(), params, S_i, state, data, priors)
        @test log_q ≈ log_q_check
    end

    @testset "MCMCOptions - New Interface" begin
        # MCMCOptions no longer has birth_proposal or assignment_method fields
        opts = MCMCOptions(n_samples=100, fixed_dim_mode=:none)
        @test opts.n_samples == 100
        @test opts.fixed_dim_mode == :none

        opts2 = MCMCOptions(fixed_dim_mode=:weighted_mean)
        @test opts2.fixed_dim_mode == :weighted_mean
    end

end
