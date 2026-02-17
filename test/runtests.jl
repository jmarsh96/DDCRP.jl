# ============================================================================
# DDCRP Test Suite
# ============================================================================

using Test
using DDCRP
using Random
using Statistics
using Distributions
using LinearAlgebra

# Set seed for reproducibility
Random.seed!(42)

# Load helper functions (not tests themselves)
include("test_helpers.jl")

@testset "DDCRP.jl" begin
    # Run actual test suites in order
    include("test_core.jl")
    include("test_proposals.jl")
    include("test_count_models.jl")
    include("test_continuous_models.jl")
    include("test_rjmcmc.jl")
    include("test_mcmc_integration.jl")
end
