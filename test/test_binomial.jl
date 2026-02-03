# ============================================================================
# Tests for Binomial Model Variants
# ============================================================================

@testset "Binomial Models" begin

    # Common setup
    Random.seed!(42)
    n = 20
    x = rand(n)
    D = construct_distance_matrix(x)
    ddcrp_params = DDCRPParams(0.1, 10.0)

    # Simulate data
    c_true = simulate_ddcrp(D; α=0.1, scale=10.0)
    tables_true = table_vector(c_true)
    p_true = [0.3, 0.6, 0.9]
    N = 10  # Number of trials

    # Assign probabilities to observations based on cluster
    p_obs = zeros(n)
    for (k, table) in enumerate(tables_true)
        prob = p_true[mod1(k, length(p_true))]
        for i in table
            p_obs[i] = prob
        end
    end
    y = rand.(Binomial.(N, p_obs))

    # ========================================================================
    # BinomialClusterProbMarg - Marginalised with Beta-Binomial conjugacy
    # ========================================================================
    @testset "BinomialClusterProbMarg" begin
        model = BinomialClusterProbMarg()
        data = CountDataWithTrials(y, N, D)
        priors = BinomialClusterProbMargPriors(2.0, 2.0)

        @testset "State Initialization" begin
            state = initialise_state(model, data, ddcrp_params, priors)

            @test state isa BinomialClusterProbMargState
            @test length(state.c) == n
        end

        @testset "Table Contribution" begin
            state = BinomialClusterProbMargState(copy(c_true))

            for table in tables_true
                contrib = table_contribution(model, table, state, data, priors)
                @test isfinite(contrib)
                @test contrib < 0  # Log-probability
            end
        end

        @testset "Posterior" begin
            state = BinomialClusterProbMargState(copy(c_true))
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, data, state, priors, log_DDCRP)
            @test isfinite(post)
        end

    end

    # ========================================================================
    # BinomialClusterProb - Unmarginalised with cluster probabilities
    # ========================================================================
    @testset "BinomialClusterProb" begin
        model = BinomialClusterProb()
        priors = BinomialClusterProbPriors(2.0, 2.0)
        data = CountDataWithTrials(y, N, D)

        @testset "State Initialization" begin
            state = initialise_state(model, data, ddcrp_params, priors)

            @test state isa BinomialClusterProbState
            @test length(state.c) == n
            @test !isempty(state.p_dict)
            @test all(0 < v < 1 for v in values(state.p_dict))
        end

        @testset "Table Contribution" begin
            tables = table_vector(c_true)
            p_dict = Dict{Vector{Int}, Float64}()
            for (k, table) in enumerate(tables)
                p_dict[sort(table)] = p_true[mod1(k, length(p_true))]
            end
            state = BinomialClusterProbState(copy(c_true), p_dict)

            for table in keys(p_dict)
                contrib = table_contribution(model, table, state, data, priors)
                @test isfinite(contrib)
            end
        end

        @testset "Posterior" begin
            tables = table_vector(c_true)
            p_dict = Dict{Vector{Int}, Float64}()
            for (k, table) in enumerate(tables)
                p_dict[sort(table)] = p_true[mod1(k, length(p_true))]
            end
            state = BinomialClusterProbState(copy(c_true), p_dict)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, data, state, priors, log_DDCRP)
            @test isfinite(post)
        end

        @testset "Cluster Parameter Update (Conjugate)" begin
            tables = table_vector(c_true)
            p_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                p_dict[sort(table)] = 0.5  # Initialize at 0.5
            end
            state = BinomialClusterProbState(copy(c_true), p_dict)

            p_before = Dict(k => v for (k, v) in state.p_dict)

            update_cluster_probs!(model, state, data, priors, tables)

            # p values should change (Gibbs update)
            @test all(0 < v < 1 for v in values(state.p_dict))
        end

    end

    # ========================================================================
    # Data Simulation
    # ========================================================================
    @testset "Data Simulation" begin
        data = simulate_binomial_data(50, 10, [0.2, 0.5, 0.8]; α=0.1, scale=10.0)

        @test length(data.y) == 50
        @test length(data.p) == 50
        @test length(data.c) == 50
        @test all(0 .<= data.y .<= 10)  # Counts within [0, N]
        @test all(0 .< data.p .< 1)     # Probabilities in (0, 1)
    end

end
