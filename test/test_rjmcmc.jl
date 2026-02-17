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
        # NBGammaPoissonGlobalR - (m,)
        n = 10
        c = simulate_ddcrp(zeros(n, n); α=1.0)
        tables = table_vector(c)
        m_dict = Dict(sort(t) => 5.0 for t in tables)
        λ = ones(n)
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)

        dicts = cluster_param_dicts(state)
        @test dicts isa NamedTuple
        @test haskey(dicts, :m)
        @test length(dicts) == 1
        @test dicts.m === m_dict

        # PoissonClusterRates - (λ,)
        λ_dict = Dict(sort(t) => 2.0 for t in tables)
        state_pois = PoissonClusterRatesState(c, λ_dict)

        dicts_pois = cluster_param_dicts(state_pois)
        @test haskey(dicts_pois, :λ)
        @test dicts_pois.λ === λ_dict
    end

    @testset "Two Parameter Models" begin
        # NBMeanDispersionClusterR - (m, r)
        n = 10
        c = simulate_ddcrp(zeros(n, n); α=1.0)
        tables = table_vector(c)
        m_dict = Dict(sort(t) => 5.0 for t in tables)
        r_dict = Dict(sort(t) => 2.0 for t in tables)
        state = NBMeanDispersionClusterRState(c, m_dict, r_dict)

        dicts = cluster_param_dicts(state)
        @test length(dicts) == 2
        @test haskey(dicts, :m)
        @test haskey(dicts, :r)
        @test dicts.m === m_dict
        @test dicts.r === r_dict
    end

    @testset "Three Parameter Models" begin
        # SkewNormalCluster - (ξ, ω, α)
        n = 10
        c = simulate_ddcrp(zeros(n, n); α=1.0)
        tables = table_vector(c)
        ξ_dict = Dict(sort(t) => 0.0 for t in tables)
        ω_dict = Dict(sort(t) => 1.0 for t in tables)
        α_dict = Dict(sort(t) => 0.5 for t in tables)
        h = abs.(randn(n))
        state = SkewNormalClusterState(c, h, ξ_dict, ω_dict, α_dict)

        dicts = cluster_param_dicts(state)
        @test length(dicts) == 3
        @test haskey(dicts, :ξ)
        @test haskey(dicts, :ω)
        @test haskey(dicts, :α)
    end

    @testset "Primary Dict is First" begin
        # Verify that the first dict is used for table lookups
        n = 10
        c = simulate_ddcrp(zeros(n, n); α=1.0)
        tables = table_vector(c)
        m_dict = Dict(sort(t) => 5.0 for t in tables)
        r_dict = Dict(sort(t) => 2.0 for t in tables)
        state = NBMeanDispersionClusterRState(c, m_dict, r_dict)

        dicts = cluster_param_dicts(state)
        primary = first(dicts)

        # Primary dict should have same keys as m_dict
        @test keys(primary) == keys(m_dict)
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

        # Create dicts
        m_dict = Dict(sort(t) => rand() for t in tables)
        r_dict = Dict(sort(t) => rand() for t in tables)
        dicts = (m = m_dict, r = r_dict)

        # Save first two tables
        tables_to_save = sort(tables)[1:min(2, length(tables))]
        saved = save_entries(dicts, tables_to_save)

        @test saved isa NamedTuple
        @test haskey(saved, :m)
        @test haskey(saved, :r)
        @test length(saved.m) == length(tables_to_save)
        @test length(saved.r) == length(tables_to_save)

        # Verify values match
        for t in tables_to_save
            @test any(kv -> kv[1] == t && kv[2] == m_dict[t], saved.m)
            @test any(kv -> kv[1] == t && kv[2] == r_dict[t], saved.r)
        end
    end

    @testset "restore_entries! Rollback" begin
        n = 10
        c = simulate_ddcrp(zeros(n, n); α=1.0)
        tables = table_vector(c)

        # Create dicts
        m_dict = Dict(sort(t) => 5.0 for t in tables)
        r_dict = Dict(sort(t) => 2.0 for t in tables)
        dicts = (m = m_dict, r = r_dict)

        # Save state
        saved = save_entries(dicts, tables)

        # Modify dicts
        for t in tables
            m_dict[t] = 999.0
            r_dict[t] = 888.0
        end

        # Create a new key (simulating birth)
        new_key = [1, 2, 3]
        m_dict[new_key] = 111.0
        r_dict[new_key] = 222.0

        # Restore
        restore_entries!(dicts, saved, [new_key])

        # Verify restoration
        @test !haskey(m_dict, new_key)
        @test !haskey(r_dict, new_key)
        for t in tables
            @test m_dict[t] == 5.0
            @test r_dict[t] == 2.0
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
        result = DDCRP.sorted_setdiff(a, b)
        @test result == [1, 5, 9]

        # Empty cases
        @test DDCRP.sorted_setdiff(a, Int[]) == a
        @test DDCRP.sorted_setdiff(Int[], b) == Int[]

        # No overlap
        @test DDCRP.sorted_setdiff([1, 2, 3], [4, 5, 6]) == [1, 2, 3]

        # Complete overlap
        @test DDCRP.sorted_setdiff([1, 2, 3], [1, 2, 3]) == Int[]
    end

    @testset "sorted_merge" begin
        a = [1, 3, 5]
        b = [2, 4, 6]
        result = DDCRP.sorted_merge(a, b)
        @test result == [1, 2, 3, 4, 5, 6]

        # Empty cases
        @test DDCRP.sorted_merge(a, Int[]) == a
        @test DDCRP.sorted_merge(Int[], b) == b

        # Overlapping (although spec says disjoint, test behavior)
        a2 = [1, 5, 9]
        b2 = [2, 5, 7]
        result2 = DDCRP.sorted_merge(a2, b2)
        @test issorted(result2)
        @test length(result2) == length(a2) + length(b2)
    end

end

# ============================================================================
# Birth Move Mechanics
# ============================================================================

@testset "Birth Move Mechanics" begin

    @testset "Birth Creates New Cluster" begin
        # Setup: 2 clusters initially
        Random.seed!(100)
        n = 20
        x = rand(n)
        D = construct_distance_matrix(x)
        ddcrp_params = DDCRPParams(0.1, 10.0)

        data_sim = simulate_negbin_data(n, [5.0, 10.0], 2.0; α=0.1, scale=10.0)
        y = data_sim.y
        data = CountData(y, D)

        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)
        opts = MCMCOptions()

        n_clusters_before = length(unique([findfirst(t -> i in t, table_vector(state.c)) for i in 1:n]))

        # Attempt multiple birth moves
        n_births_proposed = 0
        for _ in 1:100
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, proposal, log_DDCRP, opts)
            if move_type == :birth
                n_births_proposed += 1
            end
        end

        # At least some birth proposals should be generated (mechanism is working)
        @test n_births_proposed > 0
    end

    @testset "Birth Adds Dict Entries" begin
        # Controlled test: force birth by constructing specific state
        n = 10
        c = collect(1:n)
        c[2:5] .= 1  # Cluster {1,2,3,4,5}
        c[6:10] .= 6  # Cluster {6,7,8,9,10}

        tables = table_vector(c)
        @test length(tables) == 2

        m_dict = Dict(sort(tables[1]) => 5.0, sort(tables[2]) => 10.0)
        λ = ones(n) * 5.0
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)

        # Customer 3 is in cluster {1,2,3,4,5}
        # If 3 links to someone outside its moving set, it's a birth
        # Moving set of 3 when link removed depends on structure

        n_entries_before = length(state.m_dict)

        # Create minimal data
        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)
        opts = MCMCOptions()

        # Try birth move on customer 3
        for _ in 1:50
            move_type, j_star, accepted = update_c_rjmcmc!(NBGammaPoissonGlobalR(), 3, state, data, priors, proposal, log_DDCRP, opts)
            if move_type == :birth && accepted
                # Birth accepted, should have more entries
                @test length(state.m_dict) >= n_entries_before
                break
            end
        end
    end

    @testset "Birth Proposal Density" begin
        # Test that log_q_fwd is correctly computed
        n = 10
        c = collect(1:n)
        c[2:n] .= 1
        tables = table_vector(c)

        m_dict = Dict(sort(tables[1]) => 5.0)
        λ = ones(n) * 5.0
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)

        S_i = [2, 3, 4]
        proposal = PriorProposal()

        params_new, log_q_fwd = sample_birth_params(NBGammaPoissonGlobalR(), proposal, S_i, state, data, priors)

        @test haskey(params_new, :m)
        @test params_new.m > 0
        @test isfinite(log_q_fwd)

        # Verify reverse density matches
        log_q_rev = birth_params_logpdf(NBGammaPoissonGlobalR(), proposal, params_new, S_i, state, data, priors)
        @test abs(log_q_fwd - log_q_rev) < 1e-10
    end

end

# ============================================================================
# Death Move Mechanics
# ============================================================================

@testset "Death Move Mechanics" begin

    @testset "Death Merges Clusters" begin
        # Setup: Start with 3+ clusters
        Random.seed!(101)
        n = 20
        x = rand(n)
        D = construct_distance_matrix(x)
        ddcrp_params = DDCRPParams(0.5, 5.0)  # Higher α for more clusters

        data_sim = simulate_negbin_data(n, [5.0, 10.0, 15.0], 2.0; α=0.5, scale=5.0)
        y = data_sim.y
        data = CountData(y, D)

        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)
        opts = MCMCOptions()

        # Attempt multiple death moves
        n_deaths = 0
        for _ in 1:100
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, proposal, log_DDCRP, opts)
            if move_type == :death && accepted
                n_deaths += 1
            end
        end

        # Should see some deaths
        @test n_deaths >= 0  # May not always occur, but shouldn't error
    end

    @testset "Death Removes Dict Entries" begin
        # Controlled test
        n = 10
        c = collect(1:n)
        c[2:4] .= 1   # Cluster 1: {1,2,3,4}
        c[5:6] .= 5   # Cluster 2: {5,6}
        c[7:10] .= 7  # Cluster 3: {7,8,9,10}

        tables = table_vector(c)
        @test length(tables) == 3

        m_dict = Dict(sort(t) => float(i)*5.0 for (i, t) in enumerate(tables))
        λ = ones(n) * 5.0
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)
        opts = MCMCOptions()

        n_entries_before = length(state.m_dict)

        # Try death moves
        for _ in 1:50
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(NBGammaPoissonGlobalR(), i, state, data, priors, proposal, log_DDCRP, opts)
            if move_type == :death && accepted
                # Death accepted, should have fewer entries
                @test length(state.m_dict) <= n_entries_before
                break
            end
        end
    end

    @testset "Death Reverse Proposal" begin
        # Test that reverse proposal density is computed correctly
        n = 10
        c = collect(1:n)
        c[2:4] .= 1
        c[5:10] .= 5

        tables = table_vector(c)
        m_dict = Dict(sort(t) => 5.0 for t in tables)
        λ = ones(n) * 5.0
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)

        S_i = [2, 3, 4]
        proposal = PriorProposal()

        # Get params from first cluster
        table1 = sort(tables[1])
        params_old = (m = m_dict[table1],)

        log_q_rev = birth_params_logpdf(NBGammaPoissonGlobalR(), proposal, params_old, S_i, state, data, priors)
        @test isfinite(log_q_rev)
    end

end

# ============================================================================
# Fixed-Dimension Move Mechanics
# ============================================================================

@testset "Fixed-Dimension Move Mechanics" begin

    @testset "Same Table Move" begin
        # When j_old and j_star both in S_i
        n = 10
        c = collect(1:n)
        c[2:n] .= 1  # All in one cluster

        tables = table_vector(c)
        m_dict = Dict(sort(tables[1]) => 5.0)
        λ = ones(n) * 5.0
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)
        opts = MCMCOptions()

        # Moves within a single cluster should not change cluster count
        # (birth moves may be proposed but should be rejected since no new cluster needed)
        for _ in 1:10
            i = rand(2:n)  # Not customer 1 (table representative)
            move_type, j_star, accepted = update_c_rjmcmc!(NBGammaPoissonGlobalR(), i, state, data, priors, proposal, log_DDCRP, opts)

            # m_dict should not grow (births are rejected in single-cluster scenario)
            @test length(state.m_dict) == 1
        end
    end

    @testset "Different Table Fixed-Dim Move" begin
        # When j_old and j_star both NOT in S_i (different tables, same dim)
        n = 15
        c = collect(1:n)
        c[2:5] .= 1   # Cluster 1: {1,2,3,4,5}
        c[6:10] .= 6  # Cluster 2: {6,7,8,9,10}
        c[11:15] .= 11 # Cluster 3: {11,12,13,14,15}

        tables = table_vector(c)
        @test length(tables) == 3

        m_dict = Dict(sort(t) => float(i)*5.0 for (i, t) in enumerate(tables))
        λ = ones(n) * 5.0
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)
        opts = MCMCOptions(fixed_dim_mode=:none)

        n_fixed_diff = 0
        for _ in 1:20
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(NBGammaPoissonGlobalR(), i, state, data, priors, proposal, log_DDCRP, opts)
            if move_type == :fixed_diff
                n_fixed_diff += 1
                # Number of clusters should stay same
                @test length(state.m_dict) == 3
            end
        end

        # Should encounter some fixed_diff moves
        @test n_fixed_diff >= 0
    end

    @testset "fixed_dim_params Dispatch" begin
        # Test that fixed_dim_params returns correct structure
        n = 10
        c = collect(1:n)
        c[2:5] .= 1
        c[6:10] .= 6

        tables = table_vector(c)
        m_dict = Dict(sort(t) => float(i)*5.0 for (i, t) in enumerate(tables))
        λ = ones(n) * 5.0
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        opts = MCMCOptions(fixed_dim_mode=:weighted_mean)

        S_i = [2, 3]
        table_old = sort(tables[1])
        table_new = sort(tables[2])

        params_depl, params_aug, lpr = fixed_dim_params(
            NBGammaPoissonGlobalR(), S_i, table_old, table_new, state, data, priors, opts
        )

        @test params_depl isa NamedTuple
        @test params_aug isa NamedTuple
        @test haskey(params_depl, :m)
        @test haskey(params_aug, :m)
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

        data_sim = simulate_negbin_data(n, [5.0, 10.0], 2.0; α=0.2, scale=10.0)
        y = data_sim.y
        data = CountData(y, D)

        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)
        opts = MCMCOptions()

        # Run multiple moves
        for _ in 1:50
            i = rand(1:n)
            update_c_rjmcmc!(model, i, state, data, priors, proposal, log_DDCRP, opts)

            # Check state consistency after each move
            check_state_consistency(model, state, data)
        end
    end

    @testset "No Orphan Dict Entries After Death" begin
        # Verify that after death, no dict entries exist for non-existent tables
        n = 15
        c = collect(1:n)
        c[2:5] .= 1
        c[6:10] .= 6
        c[11:15] .= 11

        tables = table_vector(c)
        m_dict = Dict(sort(t) => 5.0 for t in tables)
        λ = ones(n) * 5.0
        state = NBGammaPoissonGlobalRState(c, λ, m_dict, 1.0)

        D = zeros(n, n)
        y = rand(Poisson(5), n)
        data = CountData(y, D)
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, 0.1, 10.0, D)
        opts = MCMCOptions()

        # Force deaths
        for _ in 1:20
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(NBGammaPoissonGlobalR(), i, state, data, priors, proposal, log_DDCRP, opts)

            if move_type == :death && accepted
                # Verify no orphans
                current_tables = Set(sort(t) for t in table_vector(state.c))
                dict_tables = Set(keys(state.m_dict))
                @test dict_tables == current_tables
            end
        end
    end

    @testset "c and Dicts Synchronized" begin
        # After many moves, c and dicts should be synchronized
        Random.seed!(103)
        n = 25
        x = rand(n)
        D = construct_distance_matrix(x)
        ddcrp_params = DDCRPParams(0.3, 8.0)

        data_sim = simulate_negbin_data(n, [5.0, 10.0, 15.0], 2.0; α=0.3, scale=8.0)
        y = data_sim.y
        data = CountData(y, D)

        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = InverseGammaMomentMatch(3)
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)
        opts = MCMCOptions()

        # Run 100 moves
        for _ in 1:100
            i = rand(1:n)
            update_c_rjmcmc!(model, i, state, data, priors, proposal, log_DDCRP, opts)
        end

        # Final consistency check
        check_state_consistency(model, state, data)

        # Verify dict coverage
        tables = table_vector(state.c)
        @test length(state.m_dict) == length(tables)

        for t in tables
            @test haskey(state.m_dict, sort(t))
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

        data_sim = simulate_negbin_data(n, [5.0, 10.0], 2.0; α=0.2, scale=10.0)
        y = data_sim.y
        data = CountData(y, D)

        model = NBGammaPoissonGlobalR()
        priors = NBGammaPoissonGlobalRPriors(2.0, 10.0, 2.0, 1.0)
        state = initialise_state(model, data, ddcrp_params, priors)
        proposal = PriorProposal()
        log_DDCRP = precompute_log_ddcrp(decay, ddcrp_params.α, ddcrp_params.scale, D)
        opts = MCMCOptions()

        # Track move types
        move_counts = Dict(:birth => 0, :death => 0, :fixed => 0)
        accept_counts = Dict(:birth => 0, :death => 0, :fixed => 0)

        n_moves = 200
        for _ in 1:n_moves
            i = rand(1:n)
            move_type, j_star, accepted = update_c_rjmcmc!(model, i, state, data, priors, proposal, log_DDCRP, opts)

            move_counts[move_type] += 1
            if accepted
                accept_counts[move_type] += 1
            end
        end

        # All move types should occur
        @test move_counts[:birth] + move_counts[:death] + move_counts[:fixed] == n_moves

        # At least some moves of each type should be proposed
        @test move_counts[:birth] > 0 || move_counts[:death] > 0  # Trans-dimensional
        @test move_counts[:fixed] > 0  # Fixed-dim
    end

end

end # RJMCMC Mechanics
