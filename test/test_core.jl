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
        @test ConjugateProposal <: BirthProposal
        @test NormalMomentMatch <: MomentMatchedProposal
        @test InverseGammaMomentMatch <: MomentMatchedProposal
        @test LogNormalMomentMatch <: MomentMatchedProposal
        @test FixedDistributionProposal <: BirthProposal
    end

    @testset "DDCRPParams" begin
        ddcrp = DDCRPParams(0.1, 10.0)
        @test ddcrp.α == 0.1
        @test ddcrp.scale == 10.0
        @test ddcrp.decay_fn(1.0; scale=10.0) ≈ exp(-10.0)
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
        @test opts.verbose == false
        @test opts.track_diagnostics == true

        # Test with fixed_dim_mode
        opts_fdm = MCMCOptions(fixed_dim_mode=:none)
        @test opts_fdm.fixed_dim_mode == :none

        opts_wm = MCMCOptions(fixed_dim_mode=:weighted_mean)
        @test opts_wm.fixed_dim_mode == :weighted_mean

        # Test infer_params dictionary
        opts_custom = MCMCOptions(infer_params=Dict(:λ => true, :r => false, :c => true))
        @test should_infer(opts_custom, :λ) == true
        @test should_infer(opts_custom, :r) == false
        @test should_infer(opts_custom, :c) == true
    end

    # ========================================================================
    # Edge Cases and Extended Tests
    # ========================================================================

    @testset "Edge Cases - Table Vector" begin
        # Single customer pointing to itself
        c = [1]
        tables = table_vector(c)
        @test length(tables) == 1
        @test tables[1] == [1]

        # All self-loops (n singleton clusters)
        c = [1, 2, 3, 4, 5]
        tables = table_vector(c)
        @test length(tables) == 5
        @test all(length(t) == 1 for t in tables)

        # Very long chain: 1 -> 2 -> 3 -> ... -> n -> n
        n = 20
        c = vcat(2:n, [n])
        tables = table_vector(c)
        @test length(tables) == 1
        @test sort(tables[1]) == collect(1:n)

        # Multiple disconnected components
        # Cluster 1: {1,2,3}, Cluster 2: {4,5}, Cluster 3: {6,7,8,9}
        c = [2, 3, 1, 5, 5, 7, 8, 9, 6]  # 6→7→8→9→6 forms cycle {6,7,8,9}
        tables = table_vector(c)
        @test length(tables) == 3
        @test sort(vcat(tables...)) == collect(1:9)

        # Complex cycle structure
        c = [3, 1, 2, 5, 4]  # {1,2,3} and {4,5}
        tables = table_vector(c)
        @test length(tables) == 2
    end

    @testset "Edge Cases - Decay Function" begin
        # Distance zero should give 1.0
        @test decay(0.0; scale=1.0) == 1.0
        @test decay(0.0; scale=100.0) == 1.0

        # Very large distances should approach zero
        @test decay(100.0; scale=1.0) < 1e-40
        @test decay(1000.0; scale=1.0) < 1e-300  # 1e-400 underflows to 0.0 in Float64

        # Very large scale makes decay very steep (decay(d;scale)=exp(-d*scale))
        @test decay(0.1; scale=10.0) < decay(0.1; scale=0.01)

        # Very small scale makes decay very gradual
        @test decay(10.0; scale=0.01) > 0.9  # exp(-10*0.01) = exp(-0.1) ≈ 0.905

        # Monotonicity: d1 < d2 => decay(d1) > decay(d2)
        for scale in [0.1, 1.0, 10.0]
            @test decay(0.0; scale=scale) > decay(1.0; scale=scale)
            @test decay(1.0; scale=scale) > decay(2.0; scale=scale)
            @test decay(2.0; scale=scale) > decay(10.0; scale=scale)
        end
    end

    @testset "Sorted Vector Utilities" begin
        # sorted_setdiff
        a = [1, 3, 5, 7, 9]
        b = [3, 7]
        result = sorted_setdiff(a, b)
        @test result == [1, 5, 9]
        @test issorted(result)

        # Empty sets
        @test sorted_setdiff([1, 2, 3], Int[]) == [1, 2, 3]
        @test sorted_setdiff(Int[], [1, 2, 3]) == Int[]
        @test sorted_setdiff(Int[], Int[]) == Int[]

        # Disjoint sets
        @test sorted_setdiff([1, 2], [3, 4]) == [1, 2]

        # Identical sets
        @test sorted_setdiff([1, 2, 3], [1, 2, 3]) == Int[]

        # sorted_merge
        a = [1, 3, 5]
        b = [2, 4, 6]
        result = sorted_merge(a, b)
        @test result == [1, 2, 3, 4, 5, 6]
        @test issorted(result)

        # Empty sets
        @test sorted_merge([1, 2], Int[]) == [1, 2]
        @test sorted_merge(Int[], [1, 2]) == [1, 2]
        @test sorted_merge(Int[], Int[]) == Int[]

        # Disjoint sets with interleaving (sorted_merge assumes disjoint inputs)
        a = [1, 4, 7]
        b = [3, 5, 9]
        result = sorted_merge(a, b)
        @test result == [1, 3, 4, 5, 7, 9]
        @test issorted(result)

        # Large sets (performance check)
        a_large = collect(1:2:1000)  # Odd numbers
        b_large = collect(2:2:1000)  # Even numbers
        result_large = sorted_merge(a_large, b_large)
        @test result_large == collect(1:1000)
        @test issorted(result_large)
    end

    @testset "Distance Matrix - Edge Cases" begin
        # Single point
        x = [1.0]
        D = construct_distance_matrix(x)
        @test size(D) == (1, 1)
        @test D[1, 1] == 0.0

        # Identical points (zero distances except diagonal)
        x = [5.0, 5.0, 5.0]
        D = construct_distance_matrix(x)
        @test size(D) == (3, 3)
        @test all(D .== 0.0)

        # Very large distances
        x = [0.0, 1e6]
        D = construct_distance_matrix(x)
        @test D[1, 2] == 1e6
        @test D[2, 1] == 1e6

        # Negative values
        x = [-10.0, 0.0, 10.0]
        D = construct_distance_matrix(x)
        @test D[1, 2] == 10.0
        @test D[1, 3] == 20.0
        @test D[2, 3] == 10.0
    end

    @testset "MCMCOptions - Comprehensive" begin
        # All parameters specified
        opts = MCMCOptions(
            n_samples=5000,
            verbose=true,
            infer_params=Dict(:λ => true, :r => true, :m => false, :c => true),
            prop_sds=Dict(:λ => 0.1, :r => 0.05, :m => 0.2),
            fixed_dim_mode=:resample_posterior,
            track_diagnostics=true,
            track_pairwise=true
        )

        @test opts.n_samples == 5000
        @test opts.verbose == true
        @test opts.fixed_dim_mode == :resample_posterior
        @test opts.track_diagnostics == true
        @test opts.track_pairwise == true

        # Test should_infer
        @test should_infer(opts, :λ) == true
        @test should_infer(opts, :r) == true
        @test should_infer(opts, :m) == false
        @test should_infer(opts, :c) == true

        # Test get_prop_sd
        @test get_prop_sd(opts, :λ) == 0.1
        @test get_prop_sd(opts, :r) == 0.05
        @test get_prop_sd(opts, :m) == 0.2

        # Default prop_sd when not specified
        opts_default = MCMCOptions()
        @test get_prop_sd(opts_default, :λ) > 0  # Should have some default

        # Test all fixed_dim_mode options
        for mode in [:none, :weighted_mean, :resample_posterior]
            opts_mode = MCMCOptions(fixed_dim_mode=mode)
            @test opts_mode.fixed_dim_mode == mode
        end
    end

    @testset "Cluster Label Conversion - Edge Cases" begin
        # Single observation
        c = [1]
        z = c_to_z(c, 1)
        @test length(z) == 1

        # All singletons
        c = [1, 2, 3, 4, 5]
        z = c_to_z(c, 5)
        @test length(unique(z)) == 5  # Each has unique label

        # All in one cluster
        c = [2, 1, 1, 1, 1]  # All eventually point to 1
        z = c_to_z(c, 5)
        @test length(unique(z)) == 1  # Only one cluster
    end

end
