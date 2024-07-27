# This file was generated, do not modify it. # hide
@model function negbinreg(X, y; predictors=size(X, 2))
    #priors
    α ~ Normal(0, 2.5)
    β ~ filldist(TDist(3), predictors)
    ϕ⁻ ~ Gamma(0.01, 0.01)
    ϕ = 1 / ϕ⁻

    #likelihood
    return y ~ arraydist(LazyArray(@~ NegativeBinomial2.(exp.(α .+ X * β), ϕ)))
end;