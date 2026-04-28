# ============================================================================
# Integration Tests for Full MCMC Workflow
# ============================================================================

@testset "MCMC Integration" begin

    @testset "PoissonClusterRates MCMC" begin
        Random.seed!(111)

        n = 25
        data = simulate_poisson_data(n, [5.0, 15.0]; α=0.1, scale=5.0)

        model = PoissonClusterRates()
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = PoissonClusterRatesPriors(2.0, 1.0)

        opts = MCMCOptions(
            n_samples = 300,
            verbose = false,
            track_diagnostics = true
        )

        samples, diag = mcmc(model, data.y, data.D, ddcrp_params, priors, PriorProposal(); opts=opts)

        @test samples isa AbstractMCMCSamples
        @test all(isfinite.(samples.logpost))
    end

    @testset "PoissonClusterRatesMarg MCMC" begin
        Random.seed!(222)

        n = 25
        data = simulate_poisson_data(n, [5.0, 15.0]; α=0.1, scale=5.0)

        model = PoissonClusterRatesMarg()
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = PoissonClusterRatesMargPriors(2.0, 1.0)

        opts = MCMCOptions(
            n_samples = 300,
            verbose = false,
            track_diagnostics = false
        )

        samples = mcmc(model, data.y, data.D, ddcrp_params, priors, ConjugateProposal(); opts=opts)

        @test samples isa AbstractMCMCSamples
        @test all(isfinite.(samples.logpost))
    end

    @testset "BinomialClusterProbMarg Initialization" begin
        Random.seed!(333)

        n = 25
        N = 10
        data = simulate_binomial_data(n, N, [0.3, 0.7]; α=0.1, scale=5.0)

        model = BinomialClusterProbMarg()
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = BinomialClusterProbMargPriors(2.0, 2.0)

        binom_data = CountDataWithTrials(data.y, N, data.D)
        state = initialise_state(model, binom_data, ddcrp_params, priors)
        @test state isa BinomialClusterProbMargState
    end

    @testset "New Model Variants" begin
        Random.seed!(777)

        n = 20
        data = simulate_poisson_data(n, [5.0, 10.0]; α=0.1, scale=5.0)
        ddcrp_params = DDCRPParams(0.1, 5.0)

        @testset "PoissonPopulationRates" begin
            model = PoissonPopulationRates()
            priors = PoissonPopulationRatesPriors(2.0, 1.0)

            P = rand(n) .* 1000 .+ 100
            y_pop = rand.(Poisson.(P .* 0.01))

            pop_data = CountDataWithPopulation(y_pop, round.(Int, P), data.D)
            state = initialise_state(model, pop_data, ddcrp_params, priors)
            @test state isa PoissonPopulationRatesState
            @test !isempty(state.ρ_dict)
        end
    end

    # ========================================================================
    # Continuous Model Integration Tests
    # ========================================================================

    @testset "GammaClusterShapeMarg Full MCMC" begin
        Random.seed!(888)

        n = 30
        data_sim = simulate_gamma_data(n, [2.0, 5.0], [1.0, 2.0]; α=0.1, scale=10.0)

        model = GammaClusterShapeMarg()
        ddcrp_params = DDCRPParams(0.1, 10.0)
        priors = GammaClusterShapeMargPriors(2.0, 2.0, 2.0, 1.0)

        opts = MCMCOptions(
            n_samples = 400,
            verbose = false,
            track_diagnostics = true
        )

        cont_data = ContinuousData(data_sim.y, data_sim.D)
        samples, diag = mcmc(model, cont_data, ddcrp_params, priors, PriorProposal(); opts=opts)

        @test samples isa AbstractMCMCSamples
        @test size(samples.c) == (400, n)
        @test size(samples.α) == (400, n)
        @test length(samples.logpost) == 400

        @test all(isfinite.(samples.logpost))
        @test all(samples.α .> 0)

        @test diag.birth_proposes + diag.death_proposes > 0

        ari_trace = compute_ari_trace(samples.c, data_sim.c)
        mean_ari = mean(ari_trace[201:end])
        @test_broken mean_ari >= 0.3
    end

    # ========================================================================
    # Parameter Recovery Tests
    # ========================================================================

    @testset "Parameter Recovery - GammaClusterShapeMarg" begin
        Random.seed!(1002)

        n = 40
        α_true = [2.5, 5.0]
        β_true = [1.0, 2.0]

        data_sim = simulate_gamma_data(n, α_true, β_true; α=0.15, scale=8.0)

        model = GammaClusterShapeMarg()
        ddcrp_params = DDCRPParams(0.15, 8.0)
        priors = GammaClusterShapeMargPriors(2.0, 2.0, 2.0, 1.0)

        opts = MCMCOptions(
            n_samples = 600,
            verbose = false,
            track_diagnostics = false
        )

        cont_data = ContinuousData(data_sim.y, data_sim.D)
        samples = mcmc(model, cont_data, ddcrp_params, priors, PriorProposal(); opts=opts)

        burnin = div(length(samples.logpost), 5)
        α_samples_postburn = samples.α[(burnin+1):end, :]
        mean_shapes = mean(α_samples_postburn, dims=1)[:]

        @test all(mean_shapes .> 0)

        c_recovery = test_cluster_recovery(samples.c, data_sim.c; min_ari=0.1)
        @test_broken c_recovery.mean_ari >= 0.2
    end

    @testset "Parameter Recovery - Poisson" begin
        Random.seed!(1004)

        n = 40
        λ_true = [3.0, 9.0]

        data_sim = simulate_poisson_data(n, λ_true; α=0.1, scale=10.0)

        model = PoissonClusterRates()
        ddcrp_params = DDCRPParams(0.1, 10.0)
        priors = PoissonClusterRatesPriors(2.0, 1.0)

        opts = MCMCOptions(
            n_samples = 600,
            verbose = false,
            track_diagnostics = false
        )

        samples = mcmc(model, data_sim.y, data_sim.D, ddcrp_params, priors, PriorProposal(); opts=opts)

        c_recovery = test_cluster_recovery(samples.c, data_sim.c; min_ari=0.5)
        @test c_recovery.mean_ari >= 0.3
    end

    @testset "Effective Sample Size" begin
        x = randn(1000)
        ess = effective_sample_size(x)
        @test ess > 500

        y = zeros(1000)
        y[1] = randn()
        for i in 2:1000
            y[i] = 0.9 * y[i-1] + 0.1 * randn()
        end
        ess_corr = effective_sample_size(y)
        @test ess_corr < ess
    end

    @testset "Autocorrelation" begin
        x = randn(500)
        acf = autocorrelation(x, 10)

        @test length(acf) == 11
        @test acf[1] ≈ 1.0

        @test all(abs.(acf[2:end]) .< 0.2)
    end

    @testset "Integrated Autocorrelation Time" begin
        x = randn(500)

        iat_simple = integrated_autocorrelation_time(x; method=:simple)
        iat_positive = integrated_autocorrelation_time(x; method=:initial_positive)
        iat_batch = integrated_autocorrelation_time(x; method=:batch)

        @test iat_simple >= 1.0
        @test iat_positive >= 1.0
        @test iat_batch >= 0.0
    end

end
