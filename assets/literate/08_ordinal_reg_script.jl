# This file was generated, do not modify it.

using DataFrames
using CairoMakie
using AlgebraOfGraphics
using Distributions
using StatsFuns: logit

let
    probs = [0.10, 0.15, 0.33, 0.25, 0.10, 0.07]
    dist = Categorical(probs)
    x = 1:length(probs)
    x_pmf = pdf.(dist, x)
    x_cdf = cdf.(dist, x)
    x_logodds_cdf = logit.(x_cdf)
    df = DataFrame(; x, x_pmf, x_cdf, x_logodds_cdf)
    labels = ["CDF", "Log-cumulative-odds"]
    f = Figure()
    plt1 = data(df) * mapping(:x, :x_pmf) * visual(BarPlot)
    plt2 =
        data(df) *
        mapping(:x, [:x_cdf, :x_logodds_cdf]; col=dims(1) => renamer(labels)) *
        visual(ScatterLines)
    axis = (; xticks=1:6)
    draw!(f[1, 2:3], plt1; axis)
    draw!(f[2, 1:4], plt2; axis, facet=(; linkyaxes=:none))
    f
    save(joinpath(@OUTPUT, "logodds.svg"), f) # hide
end

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

using DataFrames
using CSV
using HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/esoph.csv"
esoph = CSV.read(HTTP.get(url).body, DataFrame)

using CategoricalArrays

DataFrames.transform!(
    esoph,
    :agegp =>
        x -> categorical(
            x; levels=["25-34", "35-44", "45-54", "55-64", "65-74", "75+"], ordered=true
        ),
    :alcgp =>
        x -> categorical(x; levels=["0-39g/day", "40-79", "80-119", "120+"], ordered=true),
    :tobgp =>
        x -> categorical(x; levels=["0-9g/day", "10-19", "20-29", "30+"], ordered=true);
    renamecols=false,
)
DataFrames.transform!(
    esoph, [:agegp, :alcgp, :tobgp] .=> ByRow(levelcode); renamecols=false
)

X = Matrix(select(esoph, [:agegp, :alcgp]))
y = esoph[:, :tobgp]
model = ordreg(X, y);

chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain)

using Chain

@chain quantile(chain) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(exp); renamecols=false)
end

function logodds2prob(logodds::Float64)
    return exp(logodds) / (1 + exp(logodds))
end

@chain quantile(chain) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(logodds2prob); renamecols=false)
end
