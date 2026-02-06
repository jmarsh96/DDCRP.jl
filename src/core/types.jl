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
# Birth Proposals for RJMCMC
# ============================================================================

"""
    BirthProposal

Abstract supertype for RJMCMC birth proposal distributions.
Controls how new cluster parameters are proposed when clusters split.
Proposal objects are passed directly to `mcmc` and carry their own configuration.
"""
abstract type BirthProposal end

"""
    PriorProposal <: BirthProposal

Sample new cluster parameters from the prior distribution.
"""
struct PriorProposal <: BirthProposal end

"""
    ConjugateProposal <: BirthProposal

Marker type indicating the model has conjugate cluster parameters.
When used, `update_c!` dispatches to Gibbs sampling for assignments
instead of RJMCMC, and cluster parameters are resampled from their
conjugate posteriors after assignment updates.
"""
struct ConjugateProposal <: BirthProposal end

"""
    MomentMatchedProposal <: BirthProposal

Abstract supertype for data-informed birth proposals that use
empirical moments of the moving set to construct the proposal distribution.
"""
abstract type MomentMatchedProposal <: BirthProposal end

"""
    NormalMomentMatch <: MomentMatchedProposal

Sample new cluster parameters from truncated Normal centered at empirical mean.

# Fields
- `σ::Vector{Float64}`: One proposal std per cluster parameter
"""
struct NormalMomentMatch <: MomentMatchedProposal
    σ::Vector{Float64}
end
NormalMomentMatch(σ::Float64) = NormalMomentMatch([σ])
NormalMomentMatch(σs::Float64...) = NormalMomentMatch(collect(σs))

"""
    InverseGammaMomentMatch <: MomentMatchedProposal

Fit InverseGamma to data in moving set via method of moments.
Falls back to prior if moment matching fails.

# Fields
- `min_size::Int`: Minimum cluster size to attempt moment matching
"""
struct InverseGammaMomentMatch <: MomentMatchedProposal
    min_size::Int
end
InverseGammaMomentMatch() = InverseGammaMomentMatch(3)

"""
    LogNormalMomentMatch <: MomentMatchedProposal

Sample on log-scale using moment-matched LogNormal proposal.
For each parameter, proposes log(θ) ~ Normal(log(θ_est), σ) where
θ_est is a moment-based estimate.

# Fields
- `σ::Vector{Float64}`: One proposal std per cluster parameter (on log-scale)
- `min_size::Int`: Minimum cluster size for moment estimation
"""
struct LogNormalMomentMatch <: MomentMatchedProposal
    σ::Vector{Float64}
    min_size::Int
end
LogNormalMomentMatch(σ::Float64; min_size::Int=2) = LogNormalMomentMatch([σ], min_size)
LogNormalMomentMatch(σs::Vector{Float64}; min_size::Int=2) = LogNormalMomentMatch(σs, min_size)

"""
    FixedDistributionProposal <: BirthProposal

Sample new cluster parameters from user-specified fixed distributions.

# Fields
- `dists::Vector{UnivariateDistribution}`: One distribution per cluster parameter
"""
struct FixedDistributionProposal <: BirthProposal
    dists::Vector{UnivariateDistribution}
end
FixedDistributionProposal(d::UnivariateDistribution) = FixedDistributionProposal([d])


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
