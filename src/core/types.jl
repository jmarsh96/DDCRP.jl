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

"""
    SkewNormalModel <: LikelihoodModel

Abstract type for Skew Normal likelihood models.
Concrete subtypes: `SkewNormalCluster`
"""
abstract type SkewNormalModel <: LikelihoodModel end

"""
    GammaModel <: LikelihoodModel

Abstract type for Gamma likelihood models.
Concrete subtypes: `GammaClusterShapeMarg`
"""
abstract type GammaModel <: LikelihoodModel end

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

"""
    MomentMatchedLogNormalProposal <: BirthProposal

Sample from LogNormal centered at method-of-moments estimate.
For Gamma shape parameter α: α_est = μ²/σ² where μ, σ² are sample mean/variance.
Proposes log(α) ~ Normal(log(α_est), σ), guaranteeing positivity.

# Fields
- `σ_fixed::Float64`: Standard deviation on log-scale (default: 0.5)
- `min_size::Int`: Minimum cluster size for moment estimation (default: 2)
"""
struct MomentMatchedLogNormalProposal <: BirthProposal
    σ_fixed::Float64
    min_size::Int
end
MomentMatchedLogNormalProposal() = MomentMatchedLogNormalProposal(0.5, 2)
MomentMatchedLogNormalProposal(σ::Float64) = MomentMatchedLogNormalProposal(σ, 2)


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
Marginalised models use Gibbs sampling for customer assignments.
"""
is_marginalised(::LikelihoodModel) = false

"""
    has_latent_augmentation(model::LikelihoodModel) -> Bool

Returns true if the model uses latent augmentation variables (e.g., h_i for Skew Normal).
"""
has_latent_augmentation(::LikelihoodModel) = false

"""
    has_cluster_location(model::LikelihoodModel) -> Bool

Returns true if the model has cluster-specific location parameters (ξ_k).
"""
has_cluster_location(::LikelihoodModel) = false

"""
    has_cluster_scale(model::LikelihoodModel) -> Bool

Returns true if the model has cluster-specific scale parameters (ω_k).
"""
has_cluster_scale(::LikelihoodModel) = false

"""
    has_cluster_shape(model::LikelihoodModel) -> Bool

Returns true if the model has cluster-specific shape parameters (α_k).
"""
has_cluster_shape(::LikelihoodModel) = false

# ============================================================================
# Observed Data Types
# ============================================================================

"""
    AbstractObservedData

Abstract supertype for observed data containers in the DDCRP framework.
Encapsulates response data (y), distance matrix (D), and optional trials data.
"""
abstract type AbstractObservedData end

"""
    CountData{Ty, Td} <: AbstractObservedData

Observed count data for Poisson and Negative Binomial models.

# Fields
- `y::Ty`: Observed counts (AbstractVector)
- `D::Td`: Distance matrix (AbstractMatrix)
"""
struct CountData{Ty<:AbstractVector, Td<:AbstractMatrix} <: AbstractObservedData
    y::Ty
    D::Td
end

"""
    CountDataWithTrials{Ty, Tn, Td} <: AbstractObservedData

Observed count data with number of trials for Binomial models.

# Fields
- `y::Ty`: Observed successes (AbstractVector)
- `N::Tn`: Number of trials (scalar Int or AbstractVector{Int})
- `D::Td`: Distance matrix (AbstractMatrix)
"""
struct CountDataWithTrials{Ty<:AbstractVector, Tn<:Union{Int, <:AbstractVector{Int}}, Td<:AbstractMatrix} <: AbstractObservedData
    y::Ty
    N::Tn
    D::Td
end

"""
    ContinuousData{Ty, Td} <: AbstractObservedData

Observed continuous data for models like Skew Normal.

# Fields
- `y::Ty`: Observed values (AbstractVector{<:Real})
- `D::Td`: Distance matrix (AbstractMatrix)
"""
struct ContinuousData{Ty<:AbstractVector{<:Real}, Td<:AbstractMatrix} <: AbstractObservedData
    y::Ty
    D::Td
end

# ============================================================================
# Observed Data Accessor Functions
# ============================================================================

"""Return the observations vector."""
observations(data::AbstractObservedData) = data.y

"""Return the distance matrix."""
distance_matrix(data::AbstractObservedData) = data.D

"""Return the number of trials (only for CountDataWithTrials)."""
trials(data::CountDataWithTrials) = data.N

"""Check if data has trials information."""
has_trials(::CountData) = false
has_trials(::CountDataWithTrials) = true

"""Number of observations."""
nobs(data::AbstractObservedData) = length(data.y)

# ============================================================================
# Model Data Requirement Traits
# ============================================================================

"""
    requires_trials(model::LikelihoodModel) -> Bool

Returns true if the model requires data with trials/exposure (N or P).
"""
requires_trials(::LikelihoodModel) = false
requires_trials(::BinomialModel) = true
# Note: PoissonPopulationRates also requires trials (exposure P) - defined in model file

# ============================================================================
# Legacy Strategy Types - REMOVED
# ============================================================================
# Legacy InferenceStrategy types have been removed.
# Use model types directly with MCMCOptions to configure inference.
