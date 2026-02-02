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

@testset "DDCRP.jl" begin
    include("test_core.jl")
    include("test_proposals.jl")
    include("test_negative_binomial.jl")
    include("test_poisson.jl")
    include("test_binomial.jl")
    include("test_mcmc_integration.jl")
end
