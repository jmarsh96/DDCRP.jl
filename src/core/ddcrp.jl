# ============================================================================
# DDCRP Core Utilities
# Distance matrices, decay functions, table computation
# ============================================================================

"""
    decay(d; scale=1.0)

Exponential decay function for distance d.
Returns exp(-d * scale).
"""
decay(d; scale=1.0) = exp(-d * scale)

"""
    simulate_ddcrp(D; α=1.0, scale=1.0, decay_fn=decay)

Simulate customer assignments from the DDCRP prior.
Each customer links to another with probability proportional to distance decay,
or to themselves with probability α.
"""
function simulate_ddcrp(D; α=1.0, scale=1.0, decay_fn=decay)
    n = size(D, 1)
    customer_assignments = zeros(Int, n)
    for i in 1:n
        probs = [j != i ? decay_fn(D[i, j]; scale=scale) : α for j in 1:n]
        customer_assignments[i] = sample(1:n, Weights(probs))
    end
    return customer_assignments
end

"""
    construct_distance_matrix(x)

Construct a symmetric distance matrix D for 1D covariate vector x.
Uses absolute difference as distance metric.
"""
function construct_distance_matrix(x)
    n = length(x)
    D = zeros(n, n)
    @inbounds for i in 1:n, j in 1:n
        D[i, j] = abs(x[i] - x[j])
    end
    return D
end

"""
    precompute_log_ddcrp(f, α, scale, D)

Precompute log-DDCRP probability matrix.
- Diagonal entries: log(α) (self-link probability)
- Off-diagonal entries: log(f(D[i,j]; scale)) (distance-based link)
"""
function precompute_log_ddcrp(f, α, scale, D)
    n = size(D, 1)
    log_ddcrp = zeros(n, n)
    @inbounds for i in 1:n, j in 1:n
        if i == j
            log_ddcrp[i, j] = log(α)
        else
            log_ddcrp[i, j] = log(f(D[i, j]; scale=scale))
        end
    end
    return log_ddcrp
end

"""
    ddcrp_contribution(c, log_DDCRP)

Compute log-probability of customer assignment configuration c
using precomputed log-DDCRP matrix.
"""
function ddcrp_contribution(c, log_DDCRP)
    s = 0.0
    @inbounds for i in eachindex(c)
        s += log_DDCRP[i, c[i]]
    end
    return s
end

# Alternative signature computing log-DDCRP on the fly
function ddcrp_contribution(c, f, α, scale, D)
    sum(c[i] != i ? log(f(D[i, c[i]]; scale=scale)) : log(α) for i in eachindex(c))
end

# ============================================================================
# Table Assignment Computation
# ============================================================================

"""
    compute_table_assignments(c::Vector{Int}, force_self_loop::Int=0)

Convert customer link vector c to table assignment labels.
Uses cycle detection in the link graph.

# Arguments
- `c`: Customer assignment vector where c[i] is the customer i links to
- `force_self_loop`: If > 0, treat customer at this index as having a self-loop

# Returns
- Vector of table IDs (cluster labels) for each customer
"""
function compute_table_assignments(c::Vector{Int}, force_self_loop::Int=0)
    n = length(c)
    t_id = zeros(Int, n)  # 0 = unvisited, -1 = visiting, >0 = assigned
    path = Int[]
    sizehint!(path, n)

    next_id = 1

    for i in 1:n
        t_id[i] != 0 && continue

        # Start a new trace
        empty!(path)
        curr = i

        # Trace path until we hit a visited (-1) or assigned (>0) node
        while t_id[curr] == 0
            t_id[curr] = -1  # Mark as visiting
            push!(path, curr)
            next_node = (curr == force_self_loop) ? curr : c[curr]
            curr = next_node
        end

        # Determine table ID
        # t_id[curr] == -1 means cycle found -> new table
        # t_id[curr] > 0 means existing table -> reuse ID
        assign_id = (t_id[curr] == -1) ? (next_id += 1; next_id - 1) : t_id[curr]

        # Assign ID to all nodes in path
        for node in path
            t_id[node] = assign_id
        end
    end
    return t_id
end

"""
    table_assignments_to_vector(table_assignments)

Convert table assignment labels to vector of vectors.
Each inner vector contains indices of customers in that table.
"""
function table_assignments_to_vector(table_assignments)
    num_tables = maximum(table_assignments)
    tables = [Int[] for _ in 1:num_tables]

    for (i, table_id) in enumerate(table_assignments)
        push!(tables[table_id], i)
    end

    return tables
end

"""
    table_vector(customer_assignments)

Convert customer link vector to list of tables.
Each table is a vector of customer indices.
"""
function table_vector(customer_assignments)
    table_assignments = compute_table_assignments(customer_assignments)
    return table_assignments_to_vector(table_assignments)
end

"""
    table_vector_minus_i(i, c)

Compute table configuration after temporarily removing customer i's link.
Customer i is treated as having a self-loop.

Used in Gibbs sampling to evaluate alternative configurations.
"""
function table_vector_minus_i(i, c)
    old_c = c[i]
    c[i] = i
    table_assignments = compute_table_assignments(c)
    tables = table_assignments_to_vector(table_assignments)
    c[i] = old_c
    return tables
end

"""
    get_cluster_labels(tables, N)

Convert table list to cluster label vector.
"""
function get_cluster_labels(tables, N)
    z = zeros(Int, N)
    for (k, table) in enumerate(tables)
        for idx in table
            z[idx] = k
        end
    end
    return z
end

"""
    c_to_z(c, N)

Convert customer link vector c to cluster label vector z.
"""
function c_to_z(c, N)
    tables = table_vector(c)
    return get_cluster_labels(tables, N)
end
