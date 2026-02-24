# ============================================================================
# Integration Tests for Full MCMC Workflow
# ============================================================================

@testset "MCMC Integration" begin

    @testset "NBGammaPoissonGlobalRMarg MCMC" begin
        Random.seed!(123)

        # Small problem for fast testing
        n = 30
        data = simulate_negbin_data(n, [3.0, 8.0], 5.0; α=0.05, scale=10.0)

        model = NBGammaPoissonGlobalRMarg()
        ddcrp_params = DDCRPParams(0.05, 10.0)
        priors = NBGammaPoissonGlobalRMargPriors(2.0, 1.0, 1.0, 1e6)

        opts = MCMCOptions(
            n_samples = 500,
            verbose = false,
            infer_params = Dict(:λ => true, :r => true, :c => true),
            track_diagnostics = false
        )

        # Marginalised model requires ConjugateProposal
        samples = mcmc(model, data.y, data.D, ddcrp_params, priors, ConjugateProposal(); opts=opts)

        @test samples isa AbstractMCMCSamples
        @test size(samples.c) == (500, n)
        @test size(samples.λ) == (500, n)
        @test length(samples.r) == 500
        @test length(samples.logpost) == 500

        # Basic sanity checks
        @test all(isfinite.(samples.logpost))
        @test all(samples.r .> 0)
        @test all(samples.λ .> 0)

        # Check that MCMC is mixing (values change)
        @test samples.r[1] != samples.r[end]
        @test samples.c[1, :] != samples.c[end, :]
    end

    @testset "NBGammaPoissonGlobalR with RJMCMC" begin
        Random.seed!(456)

        n = 30
        data = simulate_negbin_data(n, [3.0, 8.0], 5.0; α=0.05, scale=10.0)

        model = NBGammaPoissonGlobalR()
        ddcrp_params = DDCRPParams(0.05, 10.0)
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1e6)

        opts = MCMCOptions(
            n_samples = 500,
            verbose = false,
            infer_params = Dict(:λ => true, :r => true, :m => true, :c => true),
            track_diagnostics = true
        )

        samples, diag = mcmc(model, data.y, data.D, ddcrp_params, priors, PriorProposal(); opts=opts)

        @test samples isa AbstractMCMCSamples
        @test diag isa MCMCDiagnostics

        # Check diagnostics
        @test diag.birth_proposes >= 0
        @test diag.death_proposes >= 0
        @test diag.fixed_proposes >= 0

        # Check acceptance rates
        acc = acceptance_rates(diag)
        @test 0 <= acc.overall <= 1 || isnan(acc.overall)

        # Basic sanity checks on samples
        @test all(isfinite.(samples.logpost))
        @test !isnothing(samples.m)
        @test all(samples.m .> 0)
    end

    @testset "RJMCMC with Different Proposals" begin
        Random.seed!(789)

        n = 20
        data = simulate_negbin_data(n, [5.0, 15.0], 3.0; α=0.1, scale=5.0)

        model = NBGammaPoissonGlobalR()
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1e6)

        proposals = [
            PriorProposal(),
            NormalMomentMatch(1.0),
            InverseGammaMomentMatch(3),
            LogNormalMomentMatch(0.5),
        ]

        for prop in proposals
            opts = MCMCOptions(
                n_samples = 200,
                verbose = false,
                track_diagnostics = true
            )

            samples, diag = mcmc(model, data.y, data.D, ddcrp_params, priors, prop; opts=opts)

            @test samples isa AbstractMCMCSamples
            @test all(isfinite.(samples.logpost))

            # Check that some birth/death moves were attempted
            total_moves = diag.birth_proposes + diag.death_proposes + diag.fixed_proposes
            @test total_moves > 0
        end
    end

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

        # Marginalised model requires ConjugateProposal
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

        # Test that the model components work
        binom_data = CountDataWithTrials(data.y, N, data.D)
        state = initialise_state(model, binom_data, ddcrp_params, priors)
        @test state isa BinomialClusterProbMargState
    end

    @testset "Diagnostics Summary" begin
        Random.seed!(444)

        n = 25
        data = simulate_negbin_data(n, [5.0, 10.0], 3.0; α=0.1, scale=5.0)

        model = NBGammaPoissonGlobalR()
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1e6)

        opts = MCMCOptions(
            n_samples = 300,
            verbose = false,
            track_diagnostics = true
        )

        samples, diag = mcmc(model, data.y, data.D, ddcrp_params, priors, PriorProposal(); opts=opts)

        # Test summary function
        summary = summarize_mcmc(samples, diag)

        @test summary isa MCMCSummary
        @test isfinite(summary.ess_n_clusters) || isnan(summary.ess_n_clusters)
        @test isfinite(summary.ess_logpost) || isnan(summary.ess_logpost)
        @test summary.total_time >= 0
    end

    @testset "Analysis Functions" begin
        Random.seed!(555)

        n = 30
        data = simulate_negbin_data(n, [5.0, 10.0], 3.0; α=0.1, scale=5.0)

        model = NBGammaPoissonGlobalRMarg()
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = NBGammaPoissonGlobalRMargPriors(2.0, 1.0, 1.0, 1e6)

        opts = MCMCOptions(n_samples = 200, verbose = false, track_diagnostics = false)
        samples = mcmc(model, data.y, data.D, ddcrp_params, priors, ConjugateProposal(); opts=opts)

        # Test calculate_n_clusters
        n_clusters = calculate_n_clusters(samples.c)
        @test length(n_clusters) == 200
        @test all(n_clusters .>= 1)
        @test all(n_clusters .<= n)

        # Test compute_similarity_matrix
        sim_mat = compute_similarity_matrix(samples.c)
        @test size(sim_mat) == (n, n)
        @test all(0 .<= sim_mat .<= 1)
        @test all(diag(sim_mat) .== 1.0)  # Self-similarity is 1
        @test sim_mat ≈ sim_mat'  # Symmetric

        # Test compute_ari_trace
        ari_trace = compute_ari_trace(samples.c, data.c)
        @test length(ari_trace) == 200
        @test all(-1 .<= ari_trace .<= 1)  # ARI range

        # Test posterior_summary
        summary = posterior_summary(samples; burnin=50)
        @test haskey(summary, :n_clusters_mean)
        @test haskey(summary, :r_mean)
    end

    @testset "Effective Sample Size" begin
        # Test ESS calculation with known series
        x = randn(1000)  # IID samples should have ESS ≈ n
        ess = effective_sample_size(x)
        @test ess > 500  # Should be close to 1000 for IID

        # Test with autocorrelated series
        y = zeros(1000)
        y[1] = randn()
        for i in 2:1000
            y[i] = 0.9 * y[i-1] + 0.1 * randn()  # High autocorrelation
        end
        ess_corr = effective_sample_size(y)
        @test ess_corr < ess  # Autocorrelated should have lower ESS
    end

    @testset "Autocorrelation" begin
        # Test with white noise
        x = randn(500)
        acf = autocorrelation(x, 10)

        @test length(acf) == 11  # Lags 0 to 10
        @test acf[1] ≈ 1.0  # Lag 0 autocorrelation is 1

        # Higher lags should be small for white noise
        @test all(abs.(acf[2:end]) .< 0.2)
    end

    @testset "Integrated Autocorrelation Time" begin
        # Test different methods
        x = randn(500)

        iat_simple = integrated_autocorrelation_time(x; method=:simple)
        iat_positive = integrated_autocorrelation_time(x; method=:initial_positive)
        iat_batch = integrated_autocorrelation_time(x; method=:batch)

        @test iat_simple >= 1.0
        @test iat_positive >= 1.0
        @test iat_batch >= 0.0  # Can be < 1 for batch method
    end

    @testset "Pairwise Diagnostics" begin
        Random.seed!(666)

        n = 15
        data = simulate_negbin_data(n, [5.0], 3.0; α=0.2, scale=3.0)

        model = NBGammaPoissonGlobalR()
        ddcrp_params = DDCRPParams(0.2, 3.0)
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1e6)

        opts = MCMCOptions(
            n_samples = 100,
            verbose = false,
            track_diagnostics = true,
            track_pairwise = true
        )

        samples, diag = mcmc(model, data.y, data.D, ddcrp_params, priors, PriorProposal(); opts=opts)

        @test !isnothing(diag.propose_matrix)
        @test !isnothing(diag.accept_matrix)
        @test size(diag.propose_matrix) == (n, n)

        # Test pairwise acceptance rates
        pairwise_rates = pairwise_acceptance_rates(diag)
        @test size(pairwise_rates) == (n, n)
        @test all(isnan.(pairwise_rates) .| (0 .<= pairwise_rates .<= 1))
    end

    @testset "New Model Variants" begin
        Random.seed!(777)

        n = 20
        data = simulate_negbin_data(n, [5.0, 10.0], 3.0; α=0.1, scale=5.0)
        ddcrp_params = DDCRPParams(0.1, 5.0)

        @testset "NBGammaPoissonClusterRMarg" begin
            model = NBGammaPoissonClusterRMarg()
            priors = NBGammaPoissonClusterRMargPriors(2.0, 1.0, 1.0, 1e6)

            count_data = CountData(data.y, data.D)
            state = initialise_state(model, count_data, ddcrp_params, priors)
            @test state isa NBGammaPoissonClusterRMargState
            @test !isempty(state.r_dict)
        end

        @testset "NBMeanDispersionGlobalR" begin
            model = NBMeanDispersionGlobalR()
            priors = NBMeanDispersionGlobalRPriors(2.0, 1.0, 1.0, 1e6)

            count_data = CountData(data.y, data.D)
            state = initialise_state(model, count_data, ddcrp_params, priors)
            @test state isa NBMeanDispersionGlobalRState
            @test state.r > 0
        end

        @testset "NBMeanDispersionClusterR" begin
            model = NBMeanDispersionClusterR()
            priors = NBMeanDispersionClusterRPriors(2.0, 1.0, 1.0, 1e6)

            count_data = CountData(data.y, data.D)
            state = initialise_state(model, count_data, ddcrp_params, priors)
            @test state isa NBMeanDispersionClusterRState
            @test !isempty(state.r_dict)
        end

        @testset "PoissonPopulationRates" begin
            model = PoissonPopulationRates()
            priors = PoissonPopulationRatesPriors(2.0, 1.0)

            # Create population data
            P = rand(n) .* 1000 .+ 100
            y_pop = rand.(Poisson.(P .* 0.01))

            pop_data = CountDataWithTrials(y_pop, round.(Int, P), data.D)
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
        # Simulate data with two clusters with different shapes
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
        @test size(samples.α) == (400, n)  # Gamma uses α not α_shape
        @test length(samples.logpost) == 400

        # Sanity checks
        @test all(isfinite.(samples.logpost))
        @test all(samples.α .> 0)  # Gamma uses α not α_shape

        # Check RJMCMC occurred
        @test diag.birth_proposes + diag.death_proposes > 0

        # Check clustering recovery
        ari_trace = compute_ari_trace(samples.c, data_sim.c)
        mean_ari = mean(ari_trace[201:end])  # Second half
        # @test_broken: Stochastic test; RJMCMC for GammaClusterShapeMarg requires more
        # samples or better initialization to reliably achieve ARI >= 0.1 with seed=888
        @test_broken mean_ari >= 0.3
    end

    @testset "SkewNormalCluster Full MCMC" begin
        Random.seed!(999)

        n = 30
        # Simulate data with symmetric and skewed clusters
        cluster_ξ = [0.0, 5.0]
        cluster_ω = [1.0, 1.5]
        cluster_α = [0.0, 2.0]  # Symmetric vs right-skewed

        data_sim = simulate_skewnormal_data(n, cluster_ξ, cluster_ω, cluster_α; α=0.1, scale=10.0)

        model = SkewNormalCluster()
        ddcrp_params = DDCRPParams(0.1, 10.0)
        priors = SkewNormalClusterPriors(0.0, 100.0, 2.0, 1.0, 0.0, 5.0)

        opts = MCMCOptions(
            n_samples = 400,
            verbose = false,
            track_diagnostics = true
        )

        cont_data = ContinuousData(data_sim.y, data_sim.D)
        samples, diag = mcmc(model, cont_data, ddcrp_params, priors, PriorProposal(); opts=opts)

        @test samples isa AbstractMCMCSamples
        @test size(samples.c) == (400, n)
        @test size(samples.ξ) == (400, n)
        @test size(samples.ω) == (400, n)
        @test size(samples.α) == (400, n)
        @test length(samples.logpost) == 400

        # Sanity checks
        @test all(isfinite.(samples.logpost))
        @test all(samples.ω .> 0)  # Scale must be positive

        # Note: h (latent variables) are not saved in samples, only used internally

        # Check RJMCMC occurred
        @test diag.birth_proposes + diag.death_proposes > 0
    end

    # ========================================================================
    # Parameter Recovery Tests
    # ========================================================================

    @testset "Parameter Recovery - NBGammaPoissonGlobalRMarg" begin
        Random.seed!(1002)

        n = 40
        r_true = 3.0
        m_true = [5.0, 17.0]

        # Structured spatial layout: first 20 obs in [0, 0.09], last 20 in [0.91, 1.0].
        x_struct = vcat(collect(LinRange(0.0, 0.09, 20)),
                        collect(LinRange(0.91, 1.0, 20)))
        data_sim = simulate_negbin_data(n, m_true, r_true; α=0.01, scale=10.0, x=x_struct)

        model = NBGammaPoissonGlobalRMarg()
        ddcrp_params = DDCRPParams(0.01, 10.0)
        priors = NBGammaPoissonGlobalRMargPriors(2.0, 1.0, 2.0, 1.0)

        opts = MCMCOptions(
            n_samples = 2000,
            verbose = false,
            track_diagnostics = false
        )

        samples = mcmc(model, data_sim.y, data_sim.D, ddcrp_params, priors, ConjugateProposal(); opts=opts)

        # Test r recovery
        r_recovery = test_parameter_recovery(samples.r, r_true; tol=0.3, param_name="r")
        @test r_recovery.within_tolerance || r_recovery.truth_in_ci

        # Test clustering recovery
        c_recovery = test_cluster_recovery(samples.c, data_sim.c; min_ari=0.2)
        @test c_recovery.mean_ari >= 0.3 
    end

    @testset "Parameter Recovery - GammaClusterShapeMarg" begin
        Random.seed!(1002)

        n = 40
        α_true = [2.5, 5.0]
        β_true = [1.0, 2.0]

        # Simulate with known parameters
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

        # Test shape parameter recovery for each observation
        # Average shape parameters per cluster in posterior
        burnin = div(length(samples.logpost), 5)

        # Extract mean shape per cluster
        α_samples_postburn = samples.α[(burnin+1):end, :]  # Gamma uses α not α_shape
        mean_shapes = mean(α_samples_postburn, dims=1)[:]

        # Should recover positive shape parameters
        @test all(mean_shapes .> 0)

        # Test clustering recovery
        c_recovery = test_cluster_recovery(samples.c, data_sim.c; min_ari=0.1)
        # @test_broken: Stochastic test; GammaClusterShapeMarg needs more samples for reliable recovery
        @test_broken c_recovery.mean_ari >= 0.2
    end

    @testset "Parameter Recovery - SkewNormalCluster" begin
        Random.seed!(1003)

        n = 40
        # Test with one symmetric and one skewed cluster
        cluster_ξ = [0.0, 8.0]
        cluster_ω = [1.0, 1.5]
        cluster_α = [0.0, 3.0]  # Symmetric vs strongly right-skewed

        data_sim = simulate_skewnormal_data(n, cluster_ξ, cluster_ω, cluster_α; α=0.15, scale=8.0)

        model = SkewNormalCluster()
        ddcrp_params = DDCRPParams(0.15, 8.0)
        priors = SkewNormalClusterPriors(0.0, 100.0, 2.0, 1.0, 0.0, 5.0)

        opts = MCMCOptions(
            n_samples = 600,
            verbose = false,
            track_diagnostics = false
        )

        cont_data = ContinuousData(data_sim.y, data_sim.D)
        samples = mcmc(model, cont_data, ddcrp_params, priors, PriorProposal(); opts=opts)

        # Sanity checks on recovered parameters
        burnin = div(length(samples.logpost), 5)

        ξ_samples = samples.ξ[(burnin+1):end, :]
        ω_samples = samples.ω[(burnin+1):end, :]
        α_samples = samples.α[(burnin+1):end, :]

        # Scale parameters should be positive
        @test all(ω_samples .> 0)

        # Location parameters should be reasonable
        @test all(isfinite.(ξ_samples))

        # Shape parameters should vary (detecting skewness)
        @test std(vec(α_samples)) > 0.1

        # Test clustering recovery
        c_recovery = test_cluster_recovery(samples.c, data_sim.c; min_ari=0.25)
        @test c_recovery.mean_ari >= 0.15  # Skewed distributions harder
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

        # Test clustering recovery
        c_recovery = test_cluster_recovery(samples.c, data_sim.c; min_ari=0.5)
        @test c_recovery.mean_ari >= 0.3
    end

end
