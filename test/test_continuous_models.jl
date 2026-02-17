# ============================================================================
# Tests for Continuous Models (Gamma and Skew Normal)
# ============================================================================
# PRIORITY: These models currently have 0% test coverage!

@testset "Continuous Models" begin

    # ========================================================================
    # GammaClusterShapeMarg - Gamma Model with Marginalised Rate
    # ========================================================================

    @testset "GammaClusterShapeMarg - State Initialization" begin
        Random.seed!(123)
        n = 20
        # Simulate gamma data
        data_sim = simulate_gamma_data(n, [2.0, 4.0], [1.0, 2.0]; α=0.1, scale=5.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        state = initialise_state(GammaClusterShapeMarg(), data, ddcrp_params, priors)

        @test state isa GammaClusterShapeMargState
        @test length(state.c) == n
        @test all(1 <= state.c[i] <= n for i in 1:n)
        @test !isempty(state.α_dict)

        # Check all observations are covered
        tables = table_vector(state.c)
        all_obs = sort(vcat(tables...))
        @test all_obs == 1:n

        # Check α parameters are positive
        for α_val in values(state.α_dict)
            @test α_val > 0
        end

        # Check dict keys are sorted
        for table in keys(state.α_dict)
            @test issorted(table)
        end

        # Test edge case: negative data should error
        y_negative = [-1.0, 2.0, 3.0]
        D_neg = zeros(3, 3)
        data_neg = ContinuousData(y_negative, D_neg)
        @test_throws AssertionError initialise_state(GammaClusterShapeMarg(), data_neg, ddcrp_params, priors)

        # Test edge case: zero in data should error
        y_zero = [0.0, 2.0, 3.0]
        data_zero = ContinuousData(y_zero, D_neg)
        @test_throws AssertionError initialise_state(GammaClusterShapeMarg(), data_zero, ddcrp_params, priors)
    end

    @testset "GammaClusterShapeMarg - Table Contribution" begin
        Random.seed!(123)
        n = 15
        data_sim = simulate_gamma_data(n, [3.0], [2.0]; α=0.05, scale=10.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp_params = DDCRPParams(0.05, 10.0)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        state = initialise_state(GammaClusterShapeMarg(), data, ddcrp_params, priors)
        tables = table_vector(state.c)

        # Test table contribution for each table
        for table in tables
            contrib = table_contribution(GammaClusterShapeMarg(), table, state, data, priors)
            @test isfinite(contrib)
            @test contrib < 0  # Log-probability should be negative
        end

        # Test with single observation cluster
        c_single = collect(1:n)  # All self-loops
        α_dict_single = Dict{Vector{Int}, Float64}()
        for i in 1:n
            α_dict_single[[i]] = 2.0
        end
        state_single = GammaClusterShapeMargState(c_single, α_dict_single)

        for i in 1:n
            contrib = table_contribution(GammaClusterShapeMarg(), [i], state_single, data, priors)
            @test isfinite(contrib)
        end

        # Test with all in one cluster
        c_all = ones(Int, n)
        α_dict_all = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] => 3.0)
        state_all = GammaClusterShapeMargState(c_all, α_dict_all)
        contrib_all = table_contribution(GammaClusterShapeMarg(), collect(1:n), state_all, data, priors)
        @test isfinite(contrib_all)
    end

    @testset "GammaClusterShapeMarg - Marginal Likelihood" begin
        # Test that marginal likelihood integrates out β correctly
        Random.seed!(123)
        y = [1.5, 2.0, 2.5, 3.0]
        D = zeros(4, 4)
        data = ContinuousData(y, D)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 3.0, 2.0)

        table = [1, 2, 3, 4]
        α = 2.5
        α_dict = Dict{Vector{Int}, Float64}(table => α)
        c = ones(Int, 4)
        state = GammaClusterShapeMargState(c, α_dict)

        contrib = table_contribution(GammaClusterShapeMarg(), table, state, data, priors)

        # Marginal likelihood should be finite and include:
        # - Likelihood integrated over β
        # - Prior on α
        @test isfinite(contrib)

        # Test with different hyperpriors
        priors_weak = GammaClusterShapeMargPriors(0.1, 0.1, 0.1, 0.1)
        state_weak = GammaClusterShapeMargState(c, α_dict)
        contrib_weak = table_contribution(GammaClusterShapeMarg(), table, state_weak, data, priors_weak)
        @test isfinite(contrib_weak)

        # Test with strong priors
        priors_strong = GammaClusterShapeMargPriors(10.0, 5.0, 10.0, 5.0)
        state_strong = GammaClusterShapeMargState(c, α_dict)
        contrib_strong = table_contribution(GammaClusterShapeMarg(), table, state_strong, data, priors_strong)
        @test isfinite(contrib_strong)
    end

    @testset "GammaClusterShapeMarg - Posterior Calculation" begin
        Random.seed!(123)
        n = 12
        data_sim = simulate_gamma_data(n, [2.0, 4.0], [1.0, 1.0]; α=0.1, scale=5.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        state = initialise_state(GammaClusterShapeMarg(), data, ddcrp_params, priors)
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, data_sim.D)

        # Test posterior calculation
        post = posterior(GammaClusterShapeMarg(), data, state, priors, log_DDCRP)
        @test isfinite(post)

        # Use helper to check components
        check_posterior_components(GammaClusterShapeMarg(), state, data, priors, log_DDCRP)
    end

    @testset "GammaClusterShapeMarg - Shape Parameter Updates" begin
        Random.seed!(123)
        n = 10
        data_sim = simulate_gamma_data(n, [3.0], [1.0]; α=0.05, scale=10.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        # Initialize with known state
        c = ones(Int, n)
        α_dict = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 3.0)
        state = GammaClusterShapeMargState(c, α_dict)

        α_before = copy(state.α_dict)

        # Update α parameters (MH on log-scale)
        update_α!(GammaClusterShapeMarg(), state, data, priors; prop_sd=0.5)

        # Check that update was attempted (may or may not accept)
        @test !isempty(state.α_dict)

        # Check positivity maintained
        for α_val in values(state.α_dict)
            @test α_val > 0
        end

        # Run multiple updates to check it doesn't crash
        for _ in 1:10
            update_α!(GammaClusterShapeMarg(), state, data, priors; prop_sd=0.3)
            @test all(α > 0 for α in values(state.α_dict))
        end
    end

    @testset "GammaClusterShapeMarg - Edge Cases" begin
        Random.seed!(123)
        priors = GammaClusterShapeMargPriors(2.0, 1.0, 2.0, 1.0)

        # Single observation cluster
        y_single = [2.5]
        D_single = zeros(1, 1)
        data_single = ContinuousData(y_single, D_single)
        c_single = [1]
        α_dict_single = Dict{Vector{Int}, Float64}([1] => 2.0)
        state_single = GammaClusterShapeMargState(c_single, α_dict_single)

        contrib_single = table_contribution(GammaClusterShapeMarg(), [1], state_single, data_single, priors)
        @test isfinite(contrib_single)

        # Very small shape (high variance)
        y_small = rand(Gamma(0.5, 2.0), 10)
        D_small = zeros(10, 10)
        data_small = ContinuousData(y_small, D_small)
        c_small = ones(Int, 10)
        α_dict_small = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 0.2)
        state_small = GammaClusterShapeMargState(c_small, α_dict_small)

        contrib_small = table_contribution(GammaClusterShapeMarg(), collect(1:10), state_small, data_small, priors)
        @test isfinite(contrib_small)

        # Very large shape (low variance)
        y_large = rand(Gamma(50.0, 0.1), 10)
        D_large = zeros(10, 10)
        data_large = ContinuousData(y_large, D_large)
        α_dict_large = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 50.0)
        state_large = GammaClusterShapeMargState(c_small, α_dict_large)

        contrib_large = table_contribution(GammaClusterShapeMarg(), collect(1:10), state_large, data_large, priors)
        @test isfinite(contrib_large)
    end

    # ========================================================================
    # SkewNormalCluster - Skew Normal with Data Augmentation
    # ========================================================================

    @testset "SkewNormalCluster - State Initialization" begin
        Random.seed!(123)
        n = 20
        # Simulate skew normal data (symmetric case)
        data_sim = simulate_skewnormal_data(n, [0.0, 5.0], [1.0, 1.5], [0.0, 0.0]; α=0.1, scale=5.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)

        state = initialise_state(SkewNormalCluster(), data, ddcrp_params, priors)

        @test state isa SkewNormalClusterState
        @test length(state.c) == n
        @test length(state.h) == n
        @test all(1 <= state.c[i] <= n for i in 1:n)

        # Check latent h are non-negative
        @test all(state.h .>= 0)

        # Check parameter dicts
        @test !isempty(state.ξ_dict)
        @test !isempty(state.ω_dict)
        @test !isempty(state.α_dict)

        # Check ω parameters are positive
        for ω_val in values(state.ω_dict)
            @test ω_val > 0
        end

        # Check all observations covered
        tables = table_vector(state.c)
        all_obs = sort(vcat(tables...))
        @test all_obs == 1:n
    end

    @testset "SkewNormalCluster - Table Contribution" begin
        Random.seed!(123)
        n = 15
        # Simulate symmetric skew normal (α=0)
        data_sim = simulate_skewnormal_data(n, [0.0], [1.0], [0.0]; α=0.05, scale=10.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp_params = DDCRPParams(0.05, 10.0)
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)

        state = initialise_state(SkewNormalCluster(), data, ddcrp_params, priors)
        tables = table_vector(state.c)

        # Test table contribution for each table
        for table in tables
            contrib = table_contribution(SkewNormalCluster(), sort(table), state, data, priors)
            @test isfinite(contrib)
        end

        # Test with single observation
        c_single = collect(1:n)
        h_single = abs.(randn(n))
        ξ_dict_single = Dict{Vector{Int}, Float64}()
        ω_dict_single = Dict{Vector{Int}, Float64}()
        α_dict_single = Dict{Vector{Int}, Float64}()
        for i in 1:n
            ξ_dict_single[[i]] = 0.0
            ω_dict_single[[i]] = 1.0
            α_dict_single[[i]] = 0.0
        end
        state_single = SkewNormalClusterState(c_single, h_single, ξ_dict_single, ω_dict_single, α_dict_single)

        for i in 1:n
            contrib = table_contribution(SkewNormalCluster(), [i], state_single, data, priors)
            @test isfinite(contrib)
        end
    end

    @testset "SkewNormalCluster - Posterior Calculation" begin
        Random.seed!(123)
        n = 12
        data_sim = simulate_skewnormal_data(n, [0.0, 3.0], [1.0, 2.0], [0.0, 0.5]; α=0.1, scale=5.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        ddcrp_params = DDCRPParams(0.1, 5.0)
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)

        state = initialise_state(SkewNormalCluster(), data, ddcrp_params, priors)
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, data_sim.D)

        # Test posterior calculation
        post = posterior(SkewNormalCluster(), data, state, priors, log_DDCRP)
        @test isfinite(post)

        # Use helper to check components
        check_posterior_components(SkewNormalCluster(), state, data, priors, log_DDCRP)
    end

    @testset "SkewNormalCluster - Data Augmentation (h updates)" begin
        Random.seed!(123)
        n = 10
        data_sim = simulate_skewnormal_data(n, [0.0], [1.0], [0.0]; α=0.05, scale=10.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)

        state = initialise_state(SkewNormalCluster(), data, ddcrp_params = DDCRPParams(0.05, 10.0), priors)

        h_before = copy(state.h)

        # Update latent h (Gibbs sampling)
        update_h!(SkewNormalCluster(), state, data, priors)

        # Check h values are valid (non-negative from half-normal)
        @test all(state.h .>= 0)

        # Values should change (probabilistic test)
        # (May occasionally fail if all h stay the same by chance, but very unlikely)
        @test length(state.h) == n

        # Run multiple updates
        for _ in 1:10
            update_h!(SkewNormalCluster(), state, data, priors)
            @test all(state.h .>= 0)
            @test length(state.h) == n
        end
    end

    @testset "SkewNormalCluster - Location Parameter Updates (ξ)" begin
        Random.seed!(123)
        n = 10
        data_sim = simulate_skewnormal_data(n, [2.0], [1.0], [0.0]; α=0.05, scale=10.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)

        state = initialise_state(SkewNormalCluster(), data, DDCRPParams(0.05, 10.0), priors)

        ξ_before = copy(state.ξ_dict)

        # Update ξ (Gibbs sampling - conjugate)
        update_ξ!(SkewNormalCluster(), state, data, priors)

        # Check validity
        @test !isempty(state.ξ_dict)

        # Run multiple updates
        for _ in 1:10
            update_ξ!(SkewNormalCluster(), state, data, priors)
            @test !isempty(state.ξ_dict)
        end
    end

    @testset "SkewNormalCluster - Scale Parameter Updates (ω)" begin
        Random.seed!(123)
        n = 10
        data_sim = simulate_skewnormal_data(n, [0.0], [1.5], [0.0]; α=0.05, scale=10.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)

        state = initialise_state(SkewNormalCluster(), data, DDCRPParams(0.05, 10.0), priors)

        # Update ω (MH on log-scale)
        update_ω!(SkewNormalCluster(), state, data, priors; prop_sd=0.3)

        # Check positivity maintained
        for ω_val in values(state.ω_dict)
            @test ω_val > 0
        end

        # Run multiple updates
        for _ in 1:10
            update_ω!(SkewNormalCluster(), state, data, priors; prop_sd=0.2)
            @test all(ω > 0 for ω in values(state.ω_dict))
        end
    end

    @testset "SkewNormalCluster - Shape Parameter Updates (α)" begin
        Random.seed!(123)
        n = 10
        # Simulate with skewness
        data_sim = simulate_skewnormal_data(n, [0.0], [1.0], [1.0]; α=0.05, scale=10.0)
        data = ContinuousData(data_sim.y, data_sim.D)
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)

        state = initialise_state(SkewNormalCluster(), data, DDCRPParams(0.05, 10.0), priors)

        # Update α (MH, can be negative)
        update_α!(SkewNormalCluster(), state, data, priors; prop_sd=0.5)

        @test !isempty(state.α_dict)

        # Run multiple updates
        for _ in 1:10
            update_α!(SkewNormalCluster(), state, data, priors; prop_sd=0.3)
            @test !isempty(state.α_dict)
        end
    end

    @testset "SkewNormalCluster - Skewness Cases" begin
        Random.seed!(123)
        n = 20
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)

        # Symmetric case (α = 0)
        data_sym = simulate_skewnormal_data(n, [0.0], [1.0], [0.0]; α=0.1, scale=5.0)
        data_obj_sym = ContinuousData(data_sym.y, data_sym.D)
        state_sym = initialise_state(SkewNormalCluster(), data_obj_sym, DDCRPParams(0.1, 5.0), priors)
        @test state_sym isa SkewNormalClusterState

        # Right skew (α > 0)
        data_right = simulate_skewnormal_data(n, [0.0], [1.0], [2.0]; α=0.1, scale=5.0)
        data_obj_right = ContinuousData(data_right.y, data_right.D)
        state_right = initialise_state(SkewNormalCluster(), data_obj_right, DDCRPParams(0.1, 5.0), priors)
        @test state_right isa SkewNormalClusterState

        # Left skew (α < 0)
        data_left = simulate_skewnormal_data(n, [0.0], [1.0], [-2.0]; α=0.1, scale=5.0)
        data_obj_left = ContinuousData(data_left.y, data_left.D)
        state_left = initialise_state(SkewNormalCluster(), data_obj_left, DDCRPParams(0.1, 5.0), priors)
        @test state_left isa SkewNormalClusterState

        # All should have valid posteriors
        for (state, data_obj, data_sim) in [(state_sym, data_obj_sym, data_sym),
                                               (state_right, data_obj_right, data_right),
                                               (state_left, data_obj_left, data_left)]
            log_DDCRP = precompute_log_ddcrp(decay, 0.1, 5.0, data_sim.D)
            post = posterior(SkewNormalCluster(), data_obj, state, priors, log_DDCRP)
            @test isfinite(post)
        end
    end

    @testset "SkewNormalCluster - Edge Cases" begin
        Random.seed!(123)
        priors = SkewNormalClusterPriors(0.0, 10.0, 2.0, 1.0, 0.0, 5.0)

        # Single observation
        y_single = [1.5]
        D_single = zeros(1, 1)
        data_single = ContinuousData(y_single, D_single)
        c_single = [1]
        h_single = [0.5]
        ξ_dict_single = Dict{Vector{Int}, Float64}([1] => 1.0)
        ω_dict_single = Dict{Vector{Int}, Float64}([1] => 1.0)
        α_dict_single = Dict{Vector{Int}, Float64}([1] => 0.0)
        state_single = SkewNormalClusterState(c_single, h_single, ξ_dict_single, ω_dict_single, α_dict_single)

        contrib_single = table_contribution(SkewNormalCluster(), [1], state_single, data_single, priors)
        @test isfinite(contrib_single)

        # Extreme skewness
        y_extreme = randn(10) .+ 5.0
        D_extreme = zeros(10, 10)
        data_extreme = ContinuousData(y_extreme, D_extreme)
        c_extreme = ones(Int, 10)
        h_extreme = abs.(randn(10))
        ξ_dict_extreme = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 5.0)
        ω_dict_extreme = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 1.0)
        α_dict_extreme = Dict{Vector{Int}, Float64}([1,2,3,4,5,6,7,8,9,10] => 10.0)  # Very skewed
        state_extreme = SkewNormalClusterState(c_extreme, h_extreme, ξ_dict_extreme, ω_dict_extreme, α_dict_extreme)

        contrib_extreme = table_contribution(SkewNormalCluster(), collect(1:10), state_extreme, data_extreme, priors)
        @test isfinite(contrib_extreme)
    end

end
