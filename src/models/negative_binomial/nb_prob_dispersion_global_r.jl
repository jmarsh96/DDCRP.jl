# ============================================================================
# NBMeanDispersionClusterR - Direct NegBin with cluster means and cluster r
# ============================================================================
#
# Model:
#   y_i | m_k, r_k ~ NegBin(r, p_k)               for observation i in cluster k
#   m_k ~ InverseGamma(m_a, m_b)                  (explicit, sampled)
#   r ~ Gamma(r_a, r_b)                         (global dispersion)
#   p_k ~ Beta(a_p, b_p)                          (cluster probabilities)
#
# Parameters: c (assignments), p_k (cluster probs), r (global dispersion)
# No latent λ rates - direct NegBin likelihood
# ============================================================================

using Distributions, SpecialFunctions, Random, Statistics

NegativeBinomial()