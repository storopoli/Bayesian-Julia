# This file was generated, do not modify it. # hide
using Turing
using Bijectors
using LazyArrays
using LinearAlgebra
using Random: seed!
using Bijectors: transformed, OrderedBijector
seed!(123)
setprogress!(false) # hide

@model function ordreg(X, y; predictors=size(X, 2), ncateg=maximum(y))
    #priors
    cutpoints ~ transformed(filldist(TDist(3) * 5, ncateg - 1), OrderedBijector())
    β ~ filldist(TDist(3) * 2.5, predictors)

    #likelihood
    return y ~ arraydist([OrderedLogistic(X[i, :] ⋅ β, cutpoints) for i in 1:length(y)])
end;