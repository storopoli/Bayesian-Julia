# This file was generated, do not modify it. # hide
using Turing
using LinearAlgebra: I
using Statistics: mean, std
using Random: seed!
seed!(123)
setprogress!(false) # hide

@model function linreg(X, y; predictors=size(X, 2))
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))
    β ~ filldist(TDist(3), predictors)
    σ ~ Exponential(1)

    #likelihood
    return y ~ MvNormal(α .+ X * β, σ^2 * I)
end;