# ============================================================================
# Count Models Test Suite
# ============================================================================

@testset "Count Models" begin

Random.seed!(43)

# ============================================================================
# Poisson Models
# ============================================================================

@testset "Poisson Models" begin

    # Common setup
    n = 25
    x = rand(n)
    D = construct_distance_matrix(x)
    ddcrp_params = DDCRPParams(0.1, 10.0)

    # Simulate Poisson data
    data_sim = simulate_poisson_data(n, [2.0, 5.0, 10.0]; α=0.1, scale=10.0)
    y = data_sim.y
    c_true = data_sim.c
    data = CountData(y, D)

    # ========================================================================
    # PoissonClusterRates - Explicit cluster rates
    # ========================================================================
    @testset "PoissonClusterRates" begin
        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)

        @testset "State Initialization" begin
            state = initialise_state(model, data, ddcrp_params, priors)

            @test state isa PoissonClusterRatesState
            @test length(state.c) == n
            @test !isempty(state.λ_dict)
            @test all(v > 0 for v in values(state.λ_dict))
        end

        @testset "Table Contribution" begin
            state = initialise_state(model, data, ddcrp_params, priors)

            for table in keys(state.λ_dict)
                contrib = table_contribution(model, table, state, data, priors)
                @test isfinite(contrib)
            end
        end

        @testset "Posterior" begin
            state = initialise_state(model, data, ddcrp_params, priors)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, data, state, priors, log_DDCRP)
            @test isfinite(post)

            check_posterior_components(model, state, data, priors, log_DDCRP)
        end

        @testset "Parameter Updates (Conjugate Gibbs)" begin
            state = initialise_state(model, data, ddcrp_params, priors)
            tables = table_vector(state.c)

            λ_before = Dict(k => v for (k, v) in state.λ_dict)
            update_cluster_rates!(model, state, data, priors, tables)

            @test all(v > 0 for v in values(state.λ_dict))
            # Rates should change (Gibbs update)
            @test any(state.λ_dict[k] != λ_before[k] for k in keys(λ_before))
        end

        @testset "Edge Cases" begin
            # Zero counts
            y_zeros = zeros(Int, 10)
            data_zeros = CountData(y_zeros, D[1:10, 1:10])
            state_zeros = initialise_state(model, data_zeros, ddcrp_params, priors)

            @test all(v > 0 for v in values(state_zeros.λ_dict))
        end
    end

    # ========================================================================
    # PoissonClusterRatesMarg - Marginalised cluster rates
    # ========================================================================
    @testset "PoissonClusterRatesMarg" begin
        model = PoissonClusterRatesMarg()
        priors = PoissonClusterRatesMargPriors(2.0, 1.0)

        @testset "State Initialization" begin
            state = initialise_state(model, data, ddcrp_params, priors)

            @test state isa PoissonClusterRatesMargState
            @test length(state.c) == n
        end

        @testset "Table Contribution (Marginal)" begin
            state = initialise_state(model, data, ddcrp_params, priors)
            tables = table_vector(state.c)

            for table in tables
                contrib = table_contribution(model, table, state, data, priors)
                @test isfinite(contrib)
                @test contrib < 0  # Log-probability
            end
        end

        @testset "Posterior" begin
            state = initialise_state(model, data, ddcrp_params, priors)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, data, state, priors, log_DDCRP)
            @test isfinite(post)
        end
    end

    # ========================================================================
    # PoissonPopulationRates - Population-specific rates with offsets
    # ========================================================================
    @testset "PoissonPopulationRates" begin
        model = PoissonPopulationRates()
        priors = PoissonPopulationRatesPriors(2.0, 1.0)

        # Create population data with integer exposures/population sizes
        E = rand(1:10, n)  # Integer population sizes
        data_pop = CountDataWithPopulation(y, E, D)

        @testset "State Initialization" begin
            state = initialise_state(model, data_pop, ddcrp_params, priors)

            @test state isa PoissonPopulationRatesState
            @test length(state.c) == n
            @test !isempty(state.ρ_dict)
            @test all(v > 0 for v in values(state.ρ_dict))
        end

        @testset "Table Contribution with Offset" begin
            state = initialise_state(model, data_pop, ddcrp_params, priors)

            for table in keys(state.ρ_dict)
                contrib = table_contribution(model, table, state, data_pop, priors)
                @test isfinite(contrib)
            end
        end

        @testset "Posterior" begin
            state = initialise_state(model, data_pop, ddcrp_params, priors)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, data_pop, state, priors, log_DDCRP)
            @test isfinite(post)
        end

        @testset "Parameter Updates" begin
            state = initialise_state(model, data_pop, ddcrp_params, priors)
            tables = table_vector(state.c)

            update_cluster_rates!(model, state, data_pop, priors, tables)
            @test all(v > 0 for v in values(state.ρ_dict))
        end
    end

    # ========================================================================
    # PoissonPopulationRatesMarg - Marginalised population-based Poisson
    # ========================================================================
    @testset "PoissonPopulationRatesMarg" begin
        model = PoissonPopulationRatesMarg()
        priors = PoissonPopulationRatesMargPriors(2.0, 1.0)

        E = rand(1:10, n)
        data_pop = CountDataWithPopulation(y, E, D)

        @testset "State Initialization" begin
            state = initialise_state(model, data_pop, ddcrp_params, priors)

            @test state isa PoissonPopulationRatesMargState
            @test length(state.c) == n
        end

        @testset "Table Contribution" begin
            state = initialise_state(model, data_pop, ddcrp_params, priors)
            tables = table_vector(state.c)

            for table in tables
                contrib = table_contribution(model, table, state, data_pop, priors)
                @test isfinite(contrib)
                @test contrib < 0
            end
        end

        @testset "Posterior" begin
            state = initialise_state(model, data_pop, ddcrp_params, priors)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, data_pop, state, priors, log_DDCRP)
            @test isfinite(post)
        end
    end

end # Poisson Models

# ============================================================================
# Binomial Models
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
            # Marginalised model - skip check_posterior_components (no cluster_param_dicts)
        end

        @testset "Edge Cases" begin
            # All successes
            y_all = fill(N, n)
            data_all = CountDataWithTrials(y_all, N, D)
            state_all = initialise_state(model, data_all, ddcrp_params, priors)
            @test length(state_all.c) == n

            # All failures
            y_none = zeros(Int, n)
            data_none = CountDataWithTrials(y_none, N, D)
            state_none = initialise_state(model, data_none, ddcrp_params, priors)
            @test length(state_none.c) == n
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

            check_posterior_components(model, state, data, priors, log_DDCRP)
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
            @test any(state.p_dict[k] != p_before[k] for k in keys(p_before))
        end

        @testset "Edge Cases" begin
            # Variable N (vector of trials)
            N_vec = rand(5:15, n)
            y_var = [rand(Binomial(N_vec[i], p_obs[i])) for i in 1:n]
            data_var = CountDataWithTrials(y_var, N_vec, D)

            state_var = initialise_state(model, data_var, ddcrp_params, priors)
            @test length(state_var.c) == n
            @test all(0 < v < 1 for v in values(state_var.p_dict))
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

end # Binomial Models

end # Count Models
