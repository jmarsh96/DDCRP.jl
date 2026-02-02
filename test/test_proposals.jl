# ============================================================================
# Tests for Birth Proposals
# ============================================================================

@testset "Birth Proposals" begin

    # Common setup - use NBGammaPoissonGlobalRPriors for testing proposals
    priors = NBGammaPoissonGlobalRPriors(2.0, 1.0, 1.0, 1.0)  # m ~ InverseGamma(2, 1)
    S_i = [1, 2, 3, 4, 5]
    λ = [2.0, 3.0, 2.5, 3.5, 2.0]

    @testset "PriorProposal" begin
        prop = PriorProposal()

        # Sample should be positive
        Random.seed!(42)
        m_new, log_q = sample_proposal(prop, S_i, λ, priors)
        @test m_new > 0
        @test isfinite(log_q)

        # log_q should match InverseGamma density
        expected_log_q = logpdf(InverseGamma(priors.m_a, priors.m_b), m_new)
        @test log_q ≈ expected_log_q

        # proposal_logpdf should be consistent
        log_q_check = proposal_logpdf(prop, m_new, S_i, λ, priors)
        @test log_q ≈ log_q_check
    end

    @testset "NormalMeanProposal" begin
        # Test with fixed σ
        prop_fixed = NormalMeanProposal(:fixed, 1.0)
        @test prop_fixed.σ_mode == :fixed
        @test prop_fixed.σ_fixed == 1.0

        Random.seed!(42)
        m_new, log_q = sample_proposal(prop_fixed, S_i, λ, priors)
        @test m_new > 0  # Truncated to positive
        @test isfinite(log_q)

        # Check log_q matches truncated normal
        μ = mean(λ[S_i])
        σ = 1.0
        expected_log_q = logpdf(truncated(Normal(μ, σ), 0.0, Inf), m_new)
        @test log_q ≈ expected_log_q

        # Test with empirical σ
        prop_emp = NormalMeanProposal(:empirical, 1.0)
        m_new2, log_q2 = sample_proposal(prop_emp, S_i, λ, priors)
        @test m_new2 > 0
        @test isfinite(log_q2)

        # Test proposal_logpdf consistency
        log_q_check = proposal_logpdf(prop_fixed, m_new, S_i, λ, priors)
        @test log_q ≈ log_q_check

        # Default constructor
        prop_default = NormalMeanProposal()
        @test prop_default.σ_mode == :empirical
    end

    @testset "MomentMatchedProposal" begin
        prop = MomentMatchedProposal(3)
        @test prop.min_size == 3

        # With sufficient data, should fit InverseGamma
        Random.seed!(42)
        m_new, log_q = sample_proposal(prop, S_i, λ, priors)
        @test m_new > 0
        @test isfinite(log_q)

        # proposal_logpdf consistency
        log_q_check = proposal_logpdf(prop, m_new, S_i, λ, priors)
        @test log_q ≈ log_q_check

        # With small cluster, should fall back to prior
        S_small = [1, 2]
        m_small, log_q_small = sample_proposal(prop, S_small, λ, priors)
        @test m_small > 0
        @test isfinite(log_q_small)

        # Default constructor
        prop_default = MomentMatchedProposal()
        @test prop_default.min_size == 3
    end

    @testset "LogNormalProposal" begin
        prop = LogNormalProposal(:empirical, 1.0)

        Random.seed!(42)
        m_new, log_q = sample_proposal(prop, S_i, λ, priors)
        @test m_new > 0
        @test isfinite(log_q)

        # Check that log_q includes Jacobian
        log_data = log.(λ[S_i])
        μ = mean(log_data)
        σ = std(log_data)
        log_m = log(m_new)
        expected_log_q = logpdf(Normal(μ, σ), log_m) - log_m
        @test log_q ≈ expected_log_q

        # proposal_logpdf consistency
        log_q_check = proposal_logpdf(prop, m_new, S_i, λ, priors)
        @test log_q ≈ log_q_check

        # Default constructor
        prop_default = LogNormalProposal()
        @test prop_default.σ_mode == :empirical
    end

    @testset "fit_inverse_gamma_moments" begin
        # Test with known data
        data = rand(InverseGamma(5.0, 10.0), 1000)
        params = fit_inverse_gamma_moments(data)

        @test !isnothing(params)
        α_fit, β_fit = params
        @test α_fit > 2  # Valid InverseGamma
        @test β_fit > 0

        # Should be close to true values for large sample
        @test abs(α_fit - 5.0) < 1.0
        @test abs(β_fit - 10.0) < 2.0

        # Test failure cases
        @test isnothing(fit_inverse_gamma_moments([1.0]))  # Too few points
        @test isnothing(fit_inverse_gamma_moments([1.0, 1.0, 1.0]))  # Zero variance
    end

    @testset "compute_proposal_σ" begin
        prop = NormalMeanProposal(:fixed, 2.0)
        @test compute_proposal_σ(prop, S_i, λ) == 2.0

        prop_emp = NormalMeanProposal(:empirical, 1.0)
        σ_emp = compute_proposal_σ(prop_emp, S_i, λ)
        @test σ_emp > 0
        @test σ_emp ≈ max(std(λ[S_i]), 0.1)

        # Single element cluster
        σ_single = compute_proposal_σ(prop_emp, [1], λ)
        @test σ_single > 0
    end

    @testset "Proposal Reversibility" begin
        # For RJMCMC, the ratio q(reverse)/q(forward) matters
        # Test that proposal_logpdf gives consistent results

        for prop in [PriorProposal(),
                     NormalMeanProposal(:empirical, 1.0),
                     MomentMatchedProposal(3),
                     LogNormalProposal(:empirical, 1.0)]
            Random.seed!(42)
            m_new, log_q_forward = sample_proposal(prop, S_i, λ, priors)
            log_q_reverse = proposal_logpdf(prop, m_new, S_i, λ, priors)

            @test log_q_forward ≈ log_q_reverse
        end
    end

    @testset "RJMCMCProposal Construction" begin
        # Test RJMCMCProposal with different birth proposals
        rj_prior = RJMCMCProposal(PriorProposal(), :none)
        @test rj_prior.birth_proposal isa PriorProposal
        @test rj_prior.fixed_dim_mode == :none

        rj_normal = RJMCMCProposal(NormalMeanProposal(), :weighted_mean)
        @test rj_normal.birth_proposal isa NormalMeanProposal
        @test rj_normal.fixed_dim_mode == :weighted_mean

        rj_moment = RJMCMCProposal(MomentMatchedProposal(5), :resample_posterior)
        @test rj_moment.birth_proposal isa MomentMatchedProposal
        @test rj_moment.fixed_dim_mode == :resample_posterior
    end

end
