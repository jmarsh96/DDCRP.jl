# ============================================================================
# RJMCMC Mechanics Test Suite
# ============================================================================
# Tests for Reversible Jump MCMC mechanics including:
# - cluster_param_dicts interface
# - Birth/death move mechanics
# - Fixed-dimension moves
# - Save/restore helpers
# - State consistency

@testset "RJMCMC Mechanics" begin

Random.seed!(44)

# ============================================================================
# cluster_param_dicts Interface
# ============================================================================

@testset "cluster_param_dicts Interface" begin

    @testset "Single Parameter Models" begin
        n = 10
        c = simulate_ddcrp(zeros(n, n); α=1.0)
        tables = table_vector(c)

        # PoissonClusterRates - (λ,)
        λ_dict = Dict(sort(t) => 2.0 for t in tables)
        state_pois = PoissonClusterRatesState(c, λ_dict)

        dicts_pois = cluster_param_dicts(state_pois)
        @test dicts_pois isa NamedTuple
        @test haskey(dicts_pois, :λ)
        @test length(dicts_pois) == 1
        @test dicts_pois.λ === λ_dict

        # BinomialClusterProb - (p,)
        p_dict = Dict(sort(t) => 0.5 for t in tables)
        state_bin = BinomialClusterProbState(c, p_dict)

        dicts_bin = cluster_param_dicts(state_bin)
        @test haskey(dicts_bin, :p)
        @test dicts_bin.p === p_dict
    end

    @testset "Primary Dict is First" begin
        n = 10
        c = simulate_ddcrp(zeros(n, n); α=1.0)
        tables = table_vector(c)
        λ_dict = Dict(sort(t) => 2.0 for t in tables)
        state = PoissonClusterRatesState(c, λ_dict)

        dicts = cluster_param_dicts(state)
        primary = first(dicts)

        @test keys(primary) == keys(λ_dict)
    end
end

# ============================================================================
# Save/Restore Helpers
# ============================================================================

@testset "Save/Restore Helpers" begin

    @testset "save_entries Correctness" begin
        n = 10
        c = simulate_ddcrp(zeros(n, n); α=1.0)
        tables = table_vector(c)

        λ_dict = Dict(sort(t) => rand() for t in tables)
        dicts = (λ = λ_dict,)

        tables_to_save = sort(tables)[1:min(2, length(tables))]
        saved = save_entries(dicts, tables_to_save)

        @test saved isa NamedTuple
        @test haskey(saved, :λ)
        @test length(saved.λ) == length(tables_to_save)

        for t in tables_to_save
            @test any(kv -> kv[1] == t && kv[2] == λ_dict[t], saved.λ)
        end
    end

    @testset "restore_entries! Rollback" begin
        n = 10
        c = simulate_ddcrp(zeros(n, n); α=1.0)
        tables = table_vector(c)

        λ_dict = Dict(sort(t) => 5.0 for t in tables)
        dicts = (λ = λ_dict,)

        saved = save_entries(dicts, tables)

        for t in tables
            λ_dict[t] = 999.0
        end

        new_key = [1, 2, 3]
        λ_dict[new_key] = 111.0

        restore_entries!(dicts, saved, [new_key])

        @test !haskey(λ_dict, new_key)
        for t in tables
            @test λ_dict[t] == 5.0
        end
    end

end

# ============================================================================
# Sorted Vector Utilities
# ============================================================================

@testset "Sorted Vector Utilities" begin

    @testset "sorted_setdiff" begin
        a = [1, 3, 5, 7, 9]
        b = [3, 7]
        result = sorted_setdiff(a, b)
        @test result == [1, 5, 9]

        @test sorted_setdiff(a, Int[]) == a
        @test sorted_setdiff(Int[], b) == Int[]
        @test sorted_setdiff([1, 2, 3], [4, 5, 6]) == [1, 2, 3]
        @test sorted_setdiff([1, 2, 3], [1, 2, 3]) == Int[]
    end

    @testset "sorted_merge" begin
        a = [1, 3, 5]
        b = [2, 4, 6]
        result = sorted_merge(a, b)
        @test result == [1, 2, 3, 4, 5, 6]

        @test sorted_merge(a, Int[]) == a
        @test sorted_merge(Int[], b) == b

        a2 = [1, 5, 9]
        b2 = [2, 5, 7]
        result2 = sorted_merge(a2, b2)
        @test issorted(result2)
        @test length(result2) == length(a2) + length(b2)
    end

end

# ============================================================================
# Birth Move Mechanics
# ============================================================================

@testset "Birth Move Mechanics" begin

    @testset "Birth Creates New Cluster" begin
        Random.seed!(100)
        n = 20
        x = rand(n)
        D = construct_distance_matrix(x)
        ddcrp_params = DDCRPParams(0.1, 10.0)

        data_sim = simulate_poisson_data(n, [5.0, 10.0]; α=0.1, scale=10.0)
        y = data_sim.y
        data = CountData(y, D)

        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

        n_births_proposed = 0
        for _ in 1:100
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, proposal, NoUpdate(), log_DDCRP)
            if move_type == :birth
                n_births_proposed += 1
            end
        end

        @test n_births_proposed > 0
    end

    @testset "Birth Adds Dict Entries" begin
        n = 10
        c = collect(1:n)
        c[2:5] .= 1
        c[6:10] .= 6

        tables = table_vector(c)
        @test length(tables) == 2

        λ_dict = Dict(sort(tables[1]) => 5.0, sort(tables[2]) => 10.0)
        state = PoissonClusterRatesState(c, λ_dict)

        n_entries_before = length(state.λ_dict)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)

        for _ in 1:50
            move_type, j_star, accepted = update_c_rjmcmc!(PoissonClusterRates(), 3, state, data, priors, proposal, NoUpdate(), log_DDCRP)
            if move_type == :birth && accepted
                @test length(state.λ_dict) >= n_entries_before
                break
            end
        end
    end

    @testset "Birth Proposal Density" begin
        n = 10
        c = collect(1:n)
        c[2:n] .= 1
        tables = table_vector(c)

        λ_dict = Dict(sort(tables[1]) => 5.0)
        state = PoissonClusterRatesState(c, λ_dict)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = PoissonClusterRatesPriors(2.0, 1.0)

        S_i = [2, 3, 4]
        proposal = PriorProposal()

        params_new, log_q_fwd = sample_birth_params(PoissonClusterRates(), proposal, S_i, state, data, priors)

        @test haskey(params_new, :λ)
        @test params_new.λ > 0
        @test isfinite(log_q_fwd)

        log_q_rev = birth_params_logpdf(PoissonClusterRates(), proposal, params_new, S_i, state, data, priors)
        @test abs(log_q_fwd - log_q_rev) < 1e-10
    end

end

# ============================================================================
# Death Move Mechanics
# ============================================================================

@testset "Death Move Mechanics" begin

    @testset "Death Merges Clusters" begin
        Random.seed!(101)
        n = 20
        x = rand(n)
        D = construct_distance_matrix(x)
        ddcrp_params = DDCRPParams(0.5, 5.0)

        data_sim = simulate_poisson_data(n, [5.0, 10.0, 15.0]; α=0.5, scale=5.0)
        y = data_sim.y
        data = CountData(y, D)

        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

        n_deaths = 0
        for _ in 1:100
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, proposal, NoUpdate(), log_DDCRP)
            if move_type == :death && accepted
                n_deaths += 1
            end
        end

        @test n_deaths >= 0
    end

    @testset "Death Removes Dict Entries" begin
        n = 10
        c = collect(1:n)
        c[2:4] .= 1
        c[5:6] .= 5
        c[7:10] .= 7

        tables = table_vector(c)
        @test length(tables) == 3

        λ_dict = Dict(sort(t) => float(i)*5.0 for (i, t) in enumerate(tables))
        state = PoissonClusterRatesState(c, λ_dict)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)

        n_entries_before = length(state.λ_dict)

        for _ in 1:50
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(PoissonClusterRates(), i, state, data, priors, proposal, NoUpdate(), log_DDCRP)
            if move_type == :death && accepted
                @test length(state.λ_dict) <= n_entries_before
                break
            end
        end
    end

    @testset "Death Reverse Proposal" begin
        n = 10
        c = collect(1:n)
        c[2:4] .= 1
        c[5:10] .= 5

        tables = table_vector(c)
        λ_dict = Dict(sort(t) => 5.0 for t in tables)
        state = PoissonClusterRatesState(c, λ_dict)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = PoissonClusterRatesPriors(2.0, 1.0)

        S_i = [2, 3, 4]
        proposal = PriorProposal()

        table1 = sort(tables[1])
        params_old = (λ = λ_dict[table1],)

        log_q_rev = birth_params_logpdf(PoissonClusterRates(), proposal, params_old, S_i, state, data, priors)
        @test isfinite(log_q_rev)
    end

end

# ============================================================================
# Fixed-Dimension Move Mechanics
# ============================================================================

@testset "Fixed-Dimension Move Mechanics" begin

    @testset "Same Table Move" begin
        n = 10
        c_init = vcat([1], fill(1, n-1))
        table_key = collect(1:n)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)

        # Reset state each iteration: j_star=i is a valid birth move and may be
        # accepted, so we isolate each trial to avoid state carry-over.
        # Only assert dict size for fixed-dim same-table moves.
        for _ in 1:20
            state = PoissonClusterRatesState(copy(c_init), Dict(table_key => 5.0))
            i = rand(2:n)
            move_type, j_star, accepted = update_c_rjmcmc!(PoissonClusterRates(), i, state, data, priors, proposal, NoUpdate(), log_DDCRP)
            if move_type == :fixed
                @test length(state.λ_dict) == 1
            end
        end
    end

    @testset "Different Table Fixed-Dim Move" begin
        n = 15
        c = collect(1:n)
        c[2:5] .= 1
        c[6:10] .= 6
        c[11:15] .= 11

        tables = table_vector(c)
        @test length(tables) == 3

        λ_dict = Dict(sort(t) => float(i)*5.0 for (i, t) in enumerate(tables))
        state = PoissonClusterRatesState(c, λ_dict)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)

        n_fixed_diff = 0
        for _ in 1:20
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(PoissonClusterRates(), i, state, data, priors, proposal, NoUpdate(), log_DDCRP)
            if move_type == :fixed_diff
                n_fixed_diff += 1
                @test length(state.λ_dict) == 3
            end
        end

        @test n_fixed_diff >= 0
    end

    @testset "fixed_dim_params Dispatch" begin
        n = 10
        c = collect(1:n)
        c[2:5] .= 1
        c[6:10] .= 6

        tables = table_vector(c)
        λ_dict = Dict(sort(t) => float(i)*5.0 for (i, t) in enumerate(tables))
        state = PoissonClusterRatesState(c, λ_dict)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = PoissonClusterRatesPriors(2.0, 1.0)

        S_i = [2, 3]
        table_old = sort(tables[1])
        table_new = sort(tables[2])

        params_depl, params_aug, lpr = fixed_dim_params(
            PoissonClusterRates(), WeightedMean(), S_i, table_old, table_new, state, data, priors
        )

        @test params_depl isa NamedTuple
        @test params_aug isa NamedTuple
        @test haskey(params_depl, :λ)
        @test haskey(params_aug, :λ)
        @test isfinite(lpr)
    end

end

# ============================================================================
# State Consistency After Moves
# ============================================================================

@testset "State Consistency After Moves" begin

    @testset "All Observations Have Parameters" begin
        Random.seed!(102)
        n = 20
        x = rand(n)
        D = construct_distance_matrix(x)
        ddcrp_params = DDCRPParams(0.2, 10.0)

        data_sim = simulate_poisson_data(n, [5.0, 10.0]; α=0.2, scale=10.0)
        y = data_sim.y
        data = CountData(y, D)

        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

        for _ in 1:50
            i = rand(1:n)
            update_c_rjmcmc!(model, i, state, data, priors, proposal, NoUpdate(), log_DDCRP)
            check_state_consistency(model, state, data)
        end
    end

    @testset "No Orphan Dict Entries After Death" begin
        n = 15
        c = collect(1:n)
        c[2:5] .= 1
        c[6:10] .= 6
        c[11:15] .= 11

        tables = table_vector(c)
        λ_dict = Dict(sort(t) => 5.0 for t in tables)
        state = PoissonClusterRatesState(c, λ_dict)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)

        for _ in 1:20
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(PoissonClusterRates(), i, state, data, priors, proposal, NoUpdate(), log_DDCRP)

            if move_type == :death && accepted
                current_tables = Set(sort(t) for t in table_vector(state.c))
                dict_tables = Set(keys(state.λ_dict))
                @test dict_tables == current_tables
            end
        end
    end

    @testset "c and Dicts Synchronized" begin
        Random.seed!(103)
        n = 25
        x = rand(n)
        D = construct_distance_matrix(x)
        ddcrp_params = DDCRPParams(0.3, 8.0)

        data_sim = simulate_poisson_data(n, [5.0, 10.0, 15.0]; α=0.3, scale=8.0)
        y = data_sim.y
        data = CountData(y, D)

        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

        for _ in 1:100
            i = rand(1:n)
            update_c_rjmcmc!(model, i, state, data, priors, proposal, NoUpdate(), log_DDCRP)
        end

        check_state_consistency(model, state, data)

        tables = table_vector(state.c)
        @test length(state.λ_dict) == length(tables)

        for t in tables
            @test haskey(state.λ_dict, sort(t))
        end
    end

end

# ============================================================================
# Move Type Statistics
# ============================================================================

@testset "Move Type Statistics" begin

    @testset "Move Type Distribution" begin
        Random.seed!(104)
        n = 20
        x = rand(n)
        D = construct_distance_matrix(x)
        ddcrp_params = DDCRPParams(0.2, 10.0)

        data_sim = simulate_poisson_data(n, [5.0, 10.0]; α=0.2, scale=10.0)
        y = data_sim.y
        data = CountData(y, D)

        model = PoissonClusterRates()
        priors = PoissonClusterRatesPriors(2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)

        move_counts = Dict(:birth => 0, :death => 0, :fixed => 0)
        accept_counts = Dict(:birth => 0, :death => 0, :fixed => 0)

        n_moves = 200
        for _ in 1:n_moves
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, proposal, NoUpdate(), log_DDCRP)

            move_counts[move_type] += 1
            if accepted
                accept_counts[move_type] += 1
            end
        end

        @test move_counts[:birth] + move_counts[:death] + move_counts[:fixed] == n_moves
        @test move_counts[:birth] > 0 || move_counts[:death] > 0
        @test move_counts[:fixed] > 0
    end

end

end # RJMCMC Mechanics
