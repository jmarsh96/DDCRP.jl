# ============================================================================
# Tests for Negative Binomial Model Variants
# ============================================================================

@testset "Negative Binomial Models" begin

    # Common setup
    Random.seed!(42)
    n = 20
    x = rand(n)
    D = construct_distance_matrix(x)
    ddcrp_params = DDCRPParams(0.1, 10.0)

    # Simulate simple data
    c_true = simulate_ddcrp(D; α=0.1, scale=10.0)
    tables_true = table_vector(c_true)
    λ_true = rand(n) .* 5 .+ 1  # λ between 1 and 6
    y = rand.(Poisson.(λ_true))

    # ========================================================================
    # NBGammaPoissonGlobalRMarg - Marginalised with global r
    # ========================================================================
    @testset "NBGammaPoissonGlobalRMarg" begin
        model = NBGammaPoissonGlobalRMarg()
        priors = NBGammaPoissonGlobalRMargPriors(2.0, 1.0, 1.0, 1e6)

        @testset "State Initialization" begin
            state = initialise_state(model, y, D, ddcrp_params, priors)

            @test state isa NBGammaPoissonGlobalRMargState
            @test length(state.c) == n
            @test length(state.λ) == n
            @test all(state.λ .> 0)
            @test state.r > 0
        end

        @testset "Table Contribution" begin
            state = NBGammaPoissonGlobalRMargState(copy(c_true), copy(λ_true), 5.0)

            for table in tables_true
                contrib = table_contribution(model, table, state, priors)
                @test isfinite(contrib)
                @test contrib < 0  # Log-probability should be negative
            end
        end

        @testset "Posterior" begin
            state = NBGammaPoissonGlobalRMargState(copy(c_true), copy(λ_true), 5.0)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, y, state, priors, log_DDCRP)
            @test isfinite(post)
        end

        @testset "Update λ" begin
            state = NBGammaPoissonGlobalRMargState(copy(c_true), copy(λ_true), 5.0)
            tables = table_vector(state.c)
            λ_before = copy(state.λ)

            for _ in 1:10
                for i in 1:n
                    update_λ!(model, i, y, state, priors, tables; prop_sd=0.5)
                end
            end

            @test state.λ != λ_before
            @test all(state.λ .> 0)
        end

        @testset "Update r" begin
            state = NBGammaPoissonGlobalRMargState(copy(c_true), copy(λ_true), 5.0)
            tables = table_vector(state.c)

            for _ in 1:20
                update_r!(model, state, priors, tables; prop_sd=0.5)
            end

            @test state.r > 0
        end

        @testset "Gibbs Customer Assignment" begin
            state = NBGammaPoissonGlobalRMargState(copy(c_true), copy(λ_true), 5.0)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)
            proposal = GibbsProposal()

            move_type, j_star, accepted = update_c!(proposal, model, 1, state, y, priors, log_DDCRP)

            @test move_type == :gibbs
            @test 1 <= j_star <= n
            @test accepted == true
        end
    end

    # ========================================================================
    # NBGammaPoissonGlobalR - Unmarginalised with global r
    # ========================================================================
    @testset "NBGammaPoissonGlobalR" begin
        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1e6)

        @testset "State Initialization" begin
            state = initialise_state(model, y, D, ddcrp_params, priors)

            @test state isa NBGammaPoissonGlobalRState
            @test length(state.c) == n
            @test length(state.λ) == n
            @test !isempty(state.m_dict)
            @test all(v > 0 for v in values(state.m_dict))
        end

        @testset "Table Contribution" begin
            tables = table_vector(c_true)
            m_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                m_dict[sort(table)] = mean(λ_true[table])
            end
            state = NBGammaPoissonGlobalRState(copy(c_true), copy(λ_true), m_dict, 5.0)

            for table in keys(m_dict)
                contrib = table_contribution(model, table, state, priors)
                @test isfinite(contrib)
            end
        end

        @testset "Posterior" begin
            tables = table_vector(c_true)
            m_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                m_dict[sort(table)] = mean(λ_true[table])
            end
            state = NBGammaPoissonGlobalRState(copy(c_true), copy(λ_true), m_dict, 5.0)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, y, state, priors, log_DDCRP)
            @test isfinite(post)
        end

        @testset "Update m" begin
            tables = table_vector(c_true)
            m_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                m_dict[sort(table)] = mean(λ_true[table])
            end
            state = NBGammaPoissonGlobalRState(copy(c_true), copy(λ_true), m_dict, 5.0)

            for _ in 1:20
                update_m!(model, state, priors; prop_sd=0.5)
            end

            @test all(v > 0 for v in values(state.m_dict))
        end

        @testset "RJMCMC Customer Assignment" begin
            tables = table_vector(c_true)
            m_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                m_dict[sort(table)] = mean(λ_true[table])
            end
            state = NBGammaPoissonGlobalRState(copy(c_true), copy(λ_true), m_dict, 5.0)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)
            proposal = RJMCMCProposal(PriorProposal(), :none)

            for _ in 1:10
                for i in 1:n
                    move_type, j_star, accepted = update_c!(proposal, model, i, state, y, priors, log_DDCRP)
                    @test move_type in [:birth, :death, :fixed]
                    @test 1 <= j_star <= n
                    @test accepted isa Bool
                end
            end

            @test all(1 <= state.c[i] <= n for i in 1:n)
            @test all(v > 0 for v in values(state.m_dict))
        end
    end

    # ========================================================================
    # NBGammaPoissonClusterRMarg - Marginalised with cluster-specific r
    # ========================================================================
    @testset "NBGammaPoissonClusterRMarg" begin
        model = NBGammaPoissonClusterRMarg()
        priors = NBGammaPoissonClusterRMargPriors(2.0, 1.0, 1.0, 1e6)

        @testset "State Initialization" begin
            state = initialise_state(model, y, D, ddcrp_params, priors)

            @test state isa NBGammaPoissonClusterRMargState
            @test length(state.c) == n
            @test length(state.λ) == n
            @test !isempty(state.r_dict)
            @test all(v > 0 for v in values(state.r_dict))
        end

        @testset "Table Contribution" begin
            tables = table_vector(c_true)
            r_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                r_dict[sort(table)] = 5.0
            end
            state = NBGammaPoissonClusterRMargState(copy(c_true), copy(λ_true), r_dict)

            for table in tables
                contrib = table_contribution(model, table, state, priors)
                @test isfinite(contrib)
            end
        end

        @testset "Posterior" begin
            tables = table_vector(c_true)
            r_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                r_dict[sort(table)] = 5.0
            end
            state = NBGammaPoissonClusterRMargState(copy(c_true), copy(λ_true), r_dict)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, y, state, priors, log_DDCRP)
            @test isfinite(post)
        end
    end

    # ========================================================================
    # NBMeanDispersionGlobalR - Direct NegBin with global r
    # ========================================================================
    @testset "NBMeanDispersionGlobalR" begin
        model = NBMeanDispersionGlobalR()
        priors = NBMeanDispersionGlobalRPriors(2.0, 1.0, 1.0, 1e6)

        @testset "State Initialization" begin
            state = initialise_state(model, y, D, ddcrp_params, priors)

            @test state isa NBMeanDispersionGlobalRState
            @test length(state.c) == n
            @test !isempty(state.m_dict)
            @test state.r > 0
        end

        @testset "Table Contribution" begin
            tables = table_vector(c_true)
            m_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                m_dict[sort(table)] = mean(y[table]) + 1.0
            end
            state = NBMeanDispersionGlobalRState(copy(c_true), m_dict, 5.0)

            for table in keys(m_dict)
                contrib = table_contribution(model, table, state, y, priors)
                @test isfinite(contrib)
            end
        end

        @testset "Posterior" begin
            tables = table_vector(c_true)
            m_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                m_dict[sort(table)] = mean(y[table]) + 1.0
            end
            state = NBMeanDispersionGlobalRState(copy(c_true), m_dict, 5.0)
            log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

            post = posterior(model, y, state, priors, log_DDCRP)
            @test isfinite(post)
        end
    end

    # ========================================================================
    # NBMeanDispersionClusterR - Direct NegBin with cluster-specific r
    # ========================================================================
    @testset "NBMeanDispersionClusterR" begin
        model = NBMeanDispersionClusterR()
        priors = NBMeanDispersionClusterRPriors(2.0, 1.0, 1.0, 1e6)

        @testset "State Initialization" begin
            state = initialise_state(model, y, D, ddcrp_params, priors)

            @test state isa NBMeanDispersionClusterRState
            @test length(state.c) == n
            @test !isempty(state.m_dict)
            @test !isempty(state.r_dict)
        end

        @testset "Table Contribution" begin
            tables = table_vector(c_true)
            m_dict = Dict{Vector{Int}, Float64}()
            r_dict = Dict{Vector{Int}, Float64}()
            for table in tables
                key = sort(table)
                m_dict[key] = mean(y[table]) + 1.0
                r_dict[key] = 5.0
            end
            state = NBMeanDispersionClusterRState(copy(c_true), m_dict, r_dict)

            for table in keys(m_dict)
                contrib = table_contribution(model, table, state, y, priors)
                @test isfinite(contrib)
            end
        end
    end

    # ========================================================================
    # Helper Functions
    # ========================================================================
    @testset "RJMCMC Helper Functions" begin
        c = [2, 1, 4, 3, 5]  # Clusters: {1,2}, {3,4}, {5}
        m_dict = Dict([1, 2] => 1.0, [3, 4] => 2.0, [5] => 3.0)

        # get_moving_set
        S_1 = get_moving_set(1, c)
        @test 1 in S_1

        # find_table_for_customer
        table_1 = find_table_for_customer(1, m_dict)
        @test table_1 == [1, 2]

        table_5 = find_table_for_customer(5, m_dict)
        @test table_5 == [5]
    end

end
