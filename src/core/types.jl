# ============================================================================
# Abstract Type Hierarchy for DDCRP Models
# ============================================================================

"""
    LikelihoodModel

Abstract supertype for all likelihood models in the DDCRP framework.
Each model variant defines how cluster data contributes to the likelihood.
"""
abstract type LikelihoodModel end

"""
    NegativeBinomialModel <: LikelihoodModel

Abstract type for Negative Binomial likelihood models.
Concrete subtypes implement specific parameterisations (Gamma-Poisson, Mean-Dispersion).
"""
abstract type NegativeBinomialModel <: LikelihoodModel end

"""
    PoissonModel <: LikelihoodModel

Abstract type for Poisson likelihood models.
Concrete subtypes: `PoissonClusterRates`, `PoissonClusterRatesMarg`, `PoissonPopulationRates`
"""
abstract type PoissonModel <: LikelihoodModel end

"""
    BinomialModel <: LikelihoodModel

Abstract type for Binomial likelihood models.
Concrete subtypes: `BinomialClusterProb`, `BinomialClusterProbMarg`
"""
abstract type BinomialModel <: LikelihoodModel end

# ============================================================================
# Abstract Types for State, Priors, and Samples
# ============================================================================

"""
    AbstractMCMCState{T<:Real}

Abstract supertype for MCMC state containers.
Each model variant has its own state type.
"""
abstract type AbstractMCMCState{T<:Real} end

"""
    AbstractPriors

Abstract supertype for prior specifications.
Allows type-dispatched prior handling and validation.
"""
abstract type AbstractPriors end

"""
    AbstractMCMCSamples

Abstract supertype for MCMC output containers.
"""
abstract type AbstractMCMCSamples end

# ============================================================================
# Birth Proposals for RJMCMC (defined first as they're used by RJMCMCProposal)
# ============================================================================

"""
    BirthProposal

Abstract supertype for RJMCMC birth proposal distributions.
Controls how new cluster parameters are proposed when clusters split.
"""
abstract type BirthProposal end

"""
    PriorProposal <: BirthProposal

Sample new cluster parameters from the prior distribution.
No Hastings correction needed when prior = proposal.
"""
struct PriorProposal <: BirthProposal end

"""
    NormalMeanProposal <: BirthProposal

Sample new cluster mean from truncated Normal centered at empirical mean.

# Fields
- `σ_mode::Symbol`: How to compute proposal std (:fixed, :empirical, :scaled)
- `σ_fixed::Float64`: Fixed value used when σ_mode == :fixed
"""
struct NormalMeanProposal <: BirthProposal
    σ_mode::Symbol
    σ_fixed::Float64
end
NormalMeanProposal() = NormalMeanProposal(:empirical, 1.0)
NormalMeanProposal(σ::Float64) = NormalMeanProposal(:fixed, σ)

"""
    MomentMatchedProposal <: BirthProposal

Fit InverseGamma to data in moving set via method of moments.
Falls back to prior if moment matching fails.

# Fields
- `min_size::Int`: Minimum cluster size to attempt moment matching
"""
struct MomentMatchedProposal <: BirthProposal
    min_size::Int
end
MomentMatchedProposal() = MomentMatchedProposal(3)

"""
    LogNormalProposal <: BirthProposal

Sample on log-scale: log(m_new) ~ Normal(μ, σ).
Appropriate when λ values span orders of magnitude.

# Fields
- `σ_mode::Symbol`: How to compute proposal std (:fixed, :empirical)
- `σ_fixed::Float64`: Fixed value used when σ_mode == :fixed
"""
struct LogNormalProposal <: BirthProposal
    σ_mode::Symbol
    σ_fixed::Float64
end
LogNormalProposal() = LogNormalProposal(:empirical, 1.0)
LogNormalProposal(σ::Float64) = LogNormalProposal(:fixed, σ)

# ============================================================================
# Assignment Proposal Types (for customer assignment updates)
# ============================================================================

"""
    AssignmentProposal

Abstract supertype for customer assignment proposal strategies.
Determines how customer assignments are updated during MCMC.
"""
abstract type AssignmentProposal end

"""
    GibbsProposal <: AssignmentProposal

Exact Gibbs sampling for customer assignments.
Only valid for marginalised models where cluster parameters are integrated out.
"""
struct GibbsProposal <: AssignmentProposal end

"""
    MetropolisProposal <: AssignmentProposal

Metropolis-Hastings proposal for customer assignments.
Proposes uniform random reassignment and accepts/rejects based on posterior ratio.
"""
struct MetropolisProposal <: AssignmentProposal end

"""
    RJMCMCProposal <: AssignmentProposal

Reversible Jump MCMC proposal for customer assignments.
Handles trans-dimensional moves (birth/death) when clusters split or merge.

# Fields
- `birth_proposal::BirthProposal`: How to propose new cluster parameters on birth moves
- `fixed_dim_mode::Symbol`: How to handle means in fixed-dimension moves (:none, :weighted_mean, :resample_posterior)
"""
struct RJMCMCProposal <: AssignmentProposal
    birth_proposal::BirthProposal
    fixed_dim_mode::Symbol
end
RJMCMCProposal(bp::BirthProposal) = RJMCMCProposal(bp, :none)
RJMCMCProposal() = RJMCMCProposal(PriorProposal(), :none)

# ============================================================================
# Model Trait Functions (to be implemented by each model variant)
# ============================================================================

"""
    has_latent_rates(model::LikelihoodModel) -> Bool

Returns true if the model uses latent observation-level rates (λ_i).
"""
has_latent_rates(::LikelihoodModel) = false

"""
    has_global_dispersion(model::LikelihoodModel) -> Bool

Returns true if the model has a global dispersion parameter (r).
"""
has_global_dispersion(::LikelihoodModel) = false

"""
    has_cluster_dispersion(model::LikelihoodModel) -> Bool

Returns true if the model has cluster-specific dispersion parameters (r_k).
"""
has_cluster_dispersion(::LikelihoodModel) = false

"""
    has_cluster_means(model::LikelihoodModel) -> Bool

Returns true if the model maintains explicit cluster mean parameters (m_k).
"""
has_cluster_means(::LikelihoodModel) = false

"""
    has_cluster_rates(model::LikelihoodModel) -> Bool

Returns true if the model maintains explicit cluster rate parameters (λ_k or ρ_k).
"""
has_cluster_rates(::LikelihoodModel) = false

"""
    has_cluster_probs(model::LikelihoodModel) -> Bool

Returns true if the model maintains explicit cluster probability parameters (p_k).
"""
has_cluster_probs(::LikelihoodModel) = false

"""
    is_marginalised(model::LikelihoodModel) -> Bool

Returns true if cluster parameters are integrated out analytically.
Models returning true can use GibbsProposal for customer assignments.
"""
is_marginalised(::LikelihoodModel) = false

"""
    default_proposal(model::LikelihoodModel) -> AssignmentProposal

Returns the default assignment proposal for the given model.
Marginalised models default to GibbsProposal, others to RJMCMCProposal.
"""
default_proposal(model::LikelihoodModel) = is_marginalised(model) ? GibbsProposal() : RJMCMCProposal()

"""
    validate_proposal(model::LikelihoodModel, proposal::AssignmentProposal)

Validates that the proposal is compatible with the model.
Throws an error if GibbsProposal is used with a non-marginalised model.
"""
function validate_proposal(model::LikelihoodModel, proposal::AssignmentProposal)
    if proposal isa GibbsProposal && !is_marginalised(model)
        throw(ArgumentError("GibbsProposal requires a marginalised model, but $(typeof(model)) is not marginalised"))
    end
end

# ============================================================================
# Legacy Strategy Types (backward compatibility)
# ============================================================================

"""
    InferenceStrategy

Legacy abstract type for inference strategies.
Kept for backward compatibility with old API.
"""
abstract type InferenceStrategy end

"""
    MarginalisedStrategy <: InferenceStrategy

Legacy abstract type for marginalised strategies.
"""
abstract type MarginalisedStrategy <: InferenceStrategy end

"""
    UnmarginalisedStrategy <: InferenceStrategy

Legacy abstract type for unmarginalised strategies.
"""
abstract type UnmarginalisedStrategy <: InferenceStrategy end

"""
    Marginalised <: MarginalisedStrategy

Legacy strategy type - use model types directly instead.
"""
struct Marginalised <: MarginalisedStrategy end

"""
    Unmarginalised <: UnmarginalisedStrategy

Legacy strategy type - use model types directly instead.
"""
struct Unmarginalised <: UnmarginalisedStrategy end

"""
    RJMCMC_Strategy <: UnmarginalisedStrategy

Legacy RJMCMC strategy - use RJMCMCProposal instead.
"""
struct RJMCMC_Strategy <: UnmarginalisedStrategy end
