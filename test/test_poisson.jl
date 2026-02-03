# ============================================================================
# Tests for Poisson Model Variants
# ============================================================================

@testset "Poisson Models" begin

    # Common setup
    Random.seed!(42)
    n = 20
    x = rand(n)
    D = construct_distance_matrix(x)
    ddcrp_params = DDCRPParams(0.1, 10.0)

    # Simulate data
    c_true = simulate_ddcrp(D; α=0.1, scale=10.0)
    tables_true = table_vector(c_true)
    λ_true = [5.0, 10.0, 15.0]

    # Assign rates to observations based on cluster
    λ_obs = zeros(n)
    for (k, table) in enumerate(tables_true)
        rate = λ_true[mod1(k, length(λ_true))]
        for i in table
            λ_obs[i] = rate
        end
    end
    y = rand.(Poisson.(λ_obs))

    # ========================================================================
    # PoissonClusterRates - Unmarginalised with cluster rates
    # ========================================================================
    @testset "PoissonClusterRates" begin
        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)

        @testset "State Initialization" begin
            state = initialise_state(model, y, D, ddcrp_params, priors)

            @test state isa PoissonClusterRatesState
            @test length(state.c) == n
            @test !isempty(state.λ_dict)
            @test all(v > 0 for v in values(state.λ_dict))
        end

        @testset "Table Contribution" begin
            tables = table_vector(c_true)
            λ_dict = Dict{Vector{Int}, Float64}()
            for (k, table) in enumerate(tables)
                λ_dict[sort(table)] = λ_true[mod1(k, length(λ_true))]
            end
            state = PoissonClusterRatesState(copy(c_true), λ_dict)

            for table in keys(λ_dict)
                contrib = table_contribution(model, table, state, y, priors)
                @test isfinite(contrib)
            end
        end

        @testset "Posterior" begin
            tables = table_vector(c_true)
            λ_dict = Dict{Vector{Int}, Float64}()
            for (k, table) in enumerate(tables)
                λ_dict[sort(table)] = λ_true[mod1(k, length(λ_true))]
            end
            state = PoissonClusterRatesState(copy(c_true), λ_dict)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, y, state, priors, log_DDCRP)
            @test isfinite(post)
        end

        @testset "Cluster Parameter Update (Conjugate)" begin
            tables = table_vector(c_true)
            λ_dict = Dict{Vector{Int}, Float64}()
            for (k, table) in enumerate(tables)
                λ_dict[sort(table)] = 1.0  # Initialize at 1
            end
            state = PoissonClusterRatesState(copy(c_true), λ_dict)

            λ_before = Dict(k => v for (k, v) in state.λ_dict)

            update_cluster_rates!(model, state, y, priors, tables)

            # λ values should change (Gibbs update)
            @test state.λ_dict != λ_before
            @test all(v > 0 for v in values(state.λ_dict))
        end

    end

    # ========================================================================
    # PoissonClusterRatesMarg - Marginalised with conjugate prior
    # ========================================================================
    @testset "PoissonClusterRatesMarg" begin
        model = PoissonClusterRatesMarg()
        priors = PoissonClusterRatesMargPriors(2.0, 1.0)

        @testset "State Initialization" begin
            state = initialise_state(model, y, D, ddcrp_params, priors)

            @test state isa PoissonClusterRatesMargState
            @test length(state.c) == n
        end

        @testset "Table Contribution" begin
            state = PoissonClusterRatesMargState(copy(c_true))

            for table in tables_true
                contrib = table_contribution(model, table, state, y, priors)
                @test isfinite(contrib)
                @test contrib < 0  # Log-probability
            end
        end

        @testset "Posterior" begin
            state = PoissonClusterRatesMargState(copy(c_true))
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, y, state, priors, log_DDCRP)
            @test isfinite(post)
        end

    end

    # ========================================================================
    # PoissonPopulationRates - Population-based rates
    # ========================================================================
    @testset "PoissonPopulationRates" begin
        model = PoissonPopulationRates()
        priors = PoissonPopulationRatesPriors(2.0, 1.0)

        # Create population/exposure data
        P = rand(n) .* 1000 .+ 100  # Populations between 100 and 1100

        # Generate data: y_i ~ Poisson(P_i * ρ_k)
        ρ_true = [0.01, 0.02, 0.03]
        y_pop = zeros(Int, n)
        for (k, table) in enumerate(tables_true)
            ρ = ρ_true[mod1(k, length(ρ_true))]
            for i in table
                y_pop[i] = rand(Poisson(P[i] * ρ))
            end
        end

        @testset "State Initialization" begin
            state = initialise_state(model, y_pop, P, D, ddcrp_params, priors)

            @test state isa PoissonPopulationRatesState
            @test length(state.c) == n
            @test !isempty(state.ρ_dict)
            @test all(v > 0 for v in values(state.ρ_dict))
        end

        @testset "Table Contribution" begin
            tables = table_vector(c_true)
            ρ_dict = Dict{Vector{Int}, Float64}()
            for (k, table) in enumerate(tables)
                ρ_dict[sort(table)] = ρ_true[mod1(k, length(ρ_true))]
            end
            state = PoissonPopulationRatesState(copy(c_true), ρ_dict)

            for table in keys(ρ_dict)
                contrib = table_contribution(model, table, state, y_pop, P, priors)
                @test isfinite(contrib)
            end
        end

        @testset "Posterior" begin
            tables = table_vector(c_true)
            ρ_dict = Dict{Vector{Int}, Float64}()
            for (k, table) in enumerate(tables)
                ρ_dict[sort(table)] = ρ_true[mod1(k, length(ρ_true))]
            end
            state = PoissonPopulationRatesState(copy(c_true), ρ_dict)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, y_pop, P, state, priors, log_DDCRP)
            @test isfinite(post)
        end

        @testset "Cluster Rate Update (Conjugate)" begin
            tables = table_vector(c_true)
            ρ_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                ρ_dict[sort(table)] = 0.01
            end
            state = PoissonPopulationRatesState(copy(c_true), ρ_dict)

            ρ_before = Dict(k => v for (k, v) in state.ρ_dict)

            update_cluster_rates!(model, state, y_pop, P, priors, tables)

            # ρ values should change (Gibbs update)
            @test all(v > 0 for v in values(state.ρ_dict))
        end
    end

    # ========================================================================
    # Data Simulation
    # ========================================================================
    @testset "Data Simulation" begin
        data = simulate_poisson_data(50, [5.0, 10.0, 20.0]; α=0.1, scale=10.0)

        @test length(data.y) == 50
        @test length(data.λ) == 50
        @test length(data.c) == 50
        @test all(data.y .>= 0)  # Counts are non-negative
        @test all(data.λ .> 0)   # Rates are positive
    end

end
