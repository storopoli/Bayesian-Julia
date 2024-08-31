# This file was generated, do not modify it. # hide
using PDMats

@model function correlated_varying_intercept_slope(
    X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)
)
    #priors
    Ω ~ LKJCholesky(predictors, 2.0) # Cholesky decomposition correlation matrix
    σ ~ Exponential(std(y))

    #prior for variance of random correlated intercepts and slopes
    #usually requires thoughtful specification
    τ ~ filldist(truncated(Cauchy(0, 2); lower=0), predictors) # group-level SDs
    γ ~ filldist(Normal(0, 5), predictors, n_gr)               # matrix of group coefficients

    #reconstruct Σ from Ω and τ
    Σ_L = Diagonal(τ) * Ω.L
    Σ = PDMat(Cholesky(Σ_L + 1e-6 * I))                        # numerical instability
    #reconstruct β from Σ and γ
    β = Σ * γ

    #likelihood
    return y ~ arraydist([Normal(X[i, :] ⋅ β[:, idx[i]], σ) for i in 1:length(y)])
end;