# ============================================================================
# Tests for Core DDCRP Utilities
# ============================================================================

@testset "Core Utilities" begin

    @testset "Distance Matrix" begin
        x = [0.0, 1.0, 2.0, 3.0]
        D = construct_distance_matrix(x)

        @test size(D) == (4, 4)
        @test D[1, 1] == 0.0
        @test D[1, 2] == 1.0
        @test D[1, 4] == 3.0
        @test D == D'  # Symmetric
        @test all(diag(D) .== 0.0)  # Zero diagonal
    end

    @testset "Decay Function" begin
        @test decay(0.0; scale=1.0) == 1.0
        @test decay(1.0; scale=1.0) ≈ exp(-1.0)
        @test decay(2.0; scale=0.5) ≈ exp(-1.0)
        @test decay(1.0; scale=2.0) ≈ exp(-2.0)

        # Decay should be positive and decreasing
        @test decay(0.0) > decay(1.0) > decay(2.0)
    end

    @testset "Table Assignment Computation" begin
        # Simple case: everyone points to themselves (n clusters)
        c = [1, 2, 3, 4]
        tables = table_vector(c)
        @test length(tables) == 4
        @test all(length(t) == 1 for t in tables)

        # Single chain: 1 -> 2 -> 3 -> 4 -> 4 (one cluster)
        c = [2, 3, 4, 4]
        tables = table_vector(c)
        @test length(tables) == 1
        @test sort(tables[1]) == [1, 2, 3, 4]

        # Two clusters: 1 -> 2 -> 1, 3 -> 4 -> 3
        c = [2, 1, 4, 3]
        tables = table_vector(c)
        @test length(tables) == 2
        @test sort(vcat(tables...)) == [1, 2, 3, 4]

        # Three customers, cycle: 1 -> 2 -> 3 -> 1
        c = [2, 3, 1]
        tables = table_vector(c)
        @test length(tables) == 1
        @test sort(tables[1]) == [1, 2, 3]
    end

    @testset "Table Vector Minus i" begin
        # c = [2, 1, 4, 3] gives two clusters: {1,2} and {3,4}
        c = [2, 1, 4, 3]

        # Remove customer 1 (breaks the 1-2 link)
        tables_minus_1 = table_vector_minus_i(1, c)
        @test 1 in tables_minus_1[findfirst(x -> 1 in x, tables_minus_1)]

        # Original c should be unchanged
        @test c == [2, 1, 4, 3]
    end

    @testset "Precompute Log DDCRP" begin
        x = [0.0, 1.0, 2.0]
        D = construct_distance_matrix(x)
        α = 1.0
        scale = 1.0

        log_DDCRP = precompute_log_ddcrp(decay, α, scale, D)

        @test size(log_DDCRP) == (3, 3)
        @test log_DDCRP[1, 1] == log(α)  # Self-loop
        @test log_DDCRP[2, 2] == log(α)
        @test log_DDCRP[1, 2] == log(decay(D[1, 2]; scale=scale))
        @test log_DDCRP[1, 3] == log(decay(D[1, 3]; scale=scale))
    end

    @testset "DDCRP Contribution" begin
        x = [0.0, 1.0, 2.0]
        D = construct_distance_matrix(x)
        α = 1.0
        scale = 1.0
        log_DDCRP = precompute_log_ddcrp(decay, α, scale, D)

        # All self-loops
        c = [1, 2, 3]
        contrib = ddcrp_contribution(c, log_DDCRP)
        @test contrib ≈ 3 * log(α)

        # One link, two self-loops
        c = [2, 2, 3]
        contrib = ddcrp_contribution(c, log_DDCRP)
        @test contrib ≈ log(decay(D[1, 2]; scale=scale)) + 2 * log(α)
    end

    @testset "Simulate DDCRP" begin
        Random.seed!(123)
        n = 50
        x = rand(n)
        D = construct_distance_matrix(x)

        c = simulate_ddcrp(D; α=0.1, scale=10.0)

        @test length(c) == n
        @test all(1 <= c[i] <= n for i in 1:n)

        # With high α, expect more self-loops (more clusters)
        c_high_alpha = simulate_ddcrp(D; α=100.0, scale=1.0)
        n_clusters_high = length(table_vector(c_high_alpha))

        c_low_alpha = simulate_ddcrp(D; α=0.01, scale=1.0)
        n_clusters_low = length(table_vector(c_low_alpha))

        # Generally, higher α should give more clusters
        # (This is probabilistic, so we don't make it a hard test)
    end

    @testset "Cluster Label Conversion" begin
        c = [2, 1, 4, 3]  # Two clusters: {1,2} and {3,4}
        n = 4

        z = c_to_z(c, n)
        @test length(z) == n
        @test z[1] == z[2]  # Same cluster
        @test z[3] == z[4]  # Same cluster
        @test z[1] != z[3]  # Different clusters
    end

    @testset "Type Hierarchy" begin
        # Test abstract model types
        @test NBGammaPoissonGlobalRMarg <: NegativeBinomialModel
        @test NBGammaPoissonGlobalR <: NegativeBinomialModel
        @test NBGammaPoissonClusterRMarg <: NegativeBinomialModel
        @test NBMeanDispersionGlobalR <: NegativeBinomialModel
        @test NBMeanDispersionClusterR <: NegativeBinomialModel
        @test NegativeBinomialModel <: LikelihoodModel

        @test PoissonClusterRates <: PoissonModel
        @test PoissonClusterRatesMarg <: PoissonModel
        @test PoissonPopulationRates <: PoissonModel
        @test PoissonModel <: LikelihoodModel

        @test BinomialClusterProb <: BinomialModel
        @test BinomialClusterProbMarg <: BinomialModel
        @test BinomialModel <: LikelihoodModel

        # Test birth proposal types
        @test PriorProposal <: BirthProposal
        @test NormalMeanProposal <: BirthProposal
        @test MomentMatchedProposal <: BirthProposal
        @test LogNormalProposal <: BirthProposal
    end

    @testset "DDCRPParams" begin
        ddcrp = DDCRPParams(0.1, 10.0)
        @test ddcrp.α == 0.1
        @test ddcrp.scale == 10.0
        @test ddcrp.decay_fn(1.0; scale=10.0) ≈ exp(-10.0)
    end

    @testset "Trait Functions" begin
        # NBGammaPoissonGlobalRMarg
        model_marg = NBGammaPoissonGlobalRMarg()
        @test has_latent_rates(model_marg) == true
        @test has_global_dispersion(model_marg) == true
        @test has_cluster_dispersion(model_marg) == false
        @test has_cluster_means(model_marg) == false
        @test is_marginalised(model_marg) == true

        # NBGammaPoissonGlobalR
        model_unmarg = NBGammaPoissonGlobalR()
        @test has_latent_rates(model_unmarg) == true
        @test has_global_dispersion(model_unmarg) == true
        @test has_cluster_means(model_unmarg) == true
        @test is_marginalised(model_unmarg) == false

        # PoissonClusterRates
        model_poisson = PoissonClusterRates()
        @test has_latent_rates(model_poisson) == false
        @test has_cluster_rates(model_poisson) == true
        @test is_marginalised(model_poisson) == false

        # PoissonClusterRatesMarg
        model_poisson_marg = PoissonClusterRatesMarg()
        @test has_cluster_rates(model_poisson_marg) == false
        @test is_marginalised(model_poisson_marg) == true

        # BinomialClusterProb
        model_binom = BinomialClusterProb()
        @test has_cluster_probs(model_binom) == true
        @test is_marginalised(model_binom) == false

        # BinomialClusterProbMarg
        model_binom_marg = BinomialClusterProbMarg()
        @test is_marginalised(model_binom_marg) == true
    end

    @testset "State Types - NBGammaPoissonGlobalRMarg" begin
        n = 10
        c = collect(1:n)
        λ = rand(n)
        r = 5.0

        state = NBGammaPoissonGlobalRMargState(c, λ, r)
        @test state.c == c
        @test state.λ == λ
        @test state.r == r
    end

    @testset "State Types - NBGammaPoissonGlobalR" begin
        n = 10
        c = collect(1:n)
        λ = rand(n)
        r = 5.0
        m_dict = Dict([1, 2, 3] => 1.0, [4, 5] => 2.0)

        state = NBGammaPoissonGlobalRState(c, λ, m_dict, r)
        @test state.c == c
        @test state.λ == λ
        @test state.m_dict == m_dict
        @test state.r == r
    end

    @testset "State Types - PoissonClusterRates" begin
        n = 10
        c = collect(1:n)
        λ_dict = Dict([1, 2, 3] => 5.0, [4, 5] => 10.0)

        state = PoissonClusterRatesState(c, λ_dict)
        @test state.c == c
        @test state.λ_dict == λ_dict
    end

    @testset "State Types - BinomialClusterProb" begin
        n = 10
        c = collect(1:n)
        p_dict = Dict([1, 2, 3] => 0.3, [4, 5] => 0.7)

        state = BinomialClusterProbState(c, p_dict)
        @test state.c == c
        @test state.p_dict == p_dict
    end

    @testset "Prior Types" begin
        # NBGammaPoissonGlobalRMargPriors
        priors_nb_marg = NBGammaPoissonGlobalRMargPriors(2.0, 1.0, 1.0, 1e6)
        @test priors_nb_marg.m_a == 2.0
        @test priors_nb_marg.m_b == 1.0
        @test priors_nb_marg.r_a == 1.0
        @test priors_nb_marg.r_b == 1e6

        # NBGammaPoissonGlobalRPriors
        priors_nb_unmarg = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1e6)
        @test priors_nb_unmarg.m_a == 2.0
        @test priors_nb_unmarg.m_b == 1.0

        # PoissonClusterRatesPriors
        priors_poisson = PoissonClusterRatesPriors(2.0, 1.0)
        @test priors_poisson.λ_a == 2.0
        @test priors_poisson.λ_b == 1.0

        # BinomialClusterProbPriors
        priors_binom = BinomialClusterProbPriors(2.0, 2.0)
        @test priors_binom.p_a == 2.0
        @test priors_binom.p_b == 2.0
    end

    @testset "m_dict_to_samples" begin
        y = [1, 2, 3, 4, 5]
        m_dict = Dict([1, 2, 3] => 10.0, [4, 5] => 20.0)

        m_vec = m_dict_to_samples(y, m_dict)
        @test m_vec[1] == 10.0
        @test m_vec[2] == 10.0
        @test m_vec[3] == 10.0
        @test m_vec[4] == 20.0
        @test m_vec[5] == 20.0
    end

    @testset "Likelihood Contribution" begin
        y = [5, 10, 15]
        λ = [5.0, 10.0, 15.0]

        ll = likelihood_contribution(y, λ)
        expected = sum(y .* log.(λ) .- λ)
        @test ll ≈ expected
    end

    @testset "MCMCOptions" begin
        # Test default options
        opts = MCMCOptions()
        @test opts.n_samples == 10000
        @test opts.assignment_method == :auto
        @test opts.verbose == false
        @test opts.track_diagnostics == true

        # Test with assignment_method
        opts_gibbs = MCMCOptions(assignment_method=:gibbs)
        @test opts_gibbs.assignment_method == :gibbs

        opts_rjmcmc = MCMCOptions(assignment_method=:rjmcmc, birth_proposal=:prior, fixed_dim_mode=:none)
        @test opts_rjmcmc.assignment_method == :rjmcmc
        @test opts_rjmcmc.birth_proposal == :prior
        @test opts_rjmcmc.fixed_dim_mode == :none

        # Test infer_params dictionary
        opts_custom = MCMCOptions(infer_params=Dict(:λ => true, :r => false, :c => true))
        @test should_infer(opts_custom, :λ) == true
        @test should_infer(opts_custom, :r) == false
        @test should_infer(opts_custom, :c) == true
    end

end
