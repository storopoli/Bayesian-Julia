# This file was generated, do not modify it.

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

using DataFrames
using CSV
using HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/kidiq.csv"
kidiq = CSV.read(HTTP.get(url).body, DataFrame)
describe(kidiq)

X = Matrix(select(kidiq, Not(:kid_score)))
y = kidiq[:, :kid_score]
model = linreg(X, y);

chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain)

quantile(chain)
