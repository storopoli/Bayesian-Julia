# This file was generated, do not modify it. # hide
using Turing
using Statistics: mean, std
using StatsBase: mad
using Random: seed!
seed!(123)
seed!(456) # hide
setprogress!(false) # hide

@model function robustreg(X, y; predictors=size(X, 2))
    #priors
    α ~ LocationScale(median(y), 2.5 * mad(y), TDist(3))
    β ~ filldist(TDist(3), predictors)
    σ ~ Exponential(1)
    ν ~ LogNormal(2, 1)

    #likelihood
    return y ~ arraydist(LocationScale.(α .+ X * β, σ, TDist.(ν)))
end;