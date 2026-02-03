# ============================================================================
# Shared Utilities for Binomial Models
# ============================================================================

"""
    logbinomial(n, k)

Compute log of binomial coefficient C(n,k) = n! / (k! * (n-k)!).
Uses loggamma for numerical stability.
"""
function logbinomial(n, k)
    return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
end
