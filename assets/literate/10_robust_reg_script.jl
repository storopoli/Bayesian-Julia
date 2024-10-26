# This file was generated, do not modify it.

using CairoMakie
using Distributions

f, ax, l = lines(-4 .. 4, Normal(0, 1); linewidth=5, axis=(; xlabel=L"x", ylabel="Density"))
save(joinpath(@OUTPUT, "normal.svg"), f); # hide

f, ax, l = lines(-4 .. 4, TDist(2); linewidth=5, axis=(xlabel=L"x", ylabel="Density"))
save(joinpath(@OUTPUT, "tdist.svg"), f); # hide

f, ax, l = lines(
    -4 .. 4,
    Normal(0, 1);
    linewidth=5,
    label="Normal",
    axis=(; xlabel=L"x", ylabel="Density"),
)
lines!(ax, -4 .. 4, TDist(2); linewidth=5, label="Student")
axislegend(ax)
save(joinpath(@OUTPUT, "comparison_normal_student.svg"), f); # hide

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

using DataFrames
using CSV
using HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/duncan.csv"
duncan = CSV.read(HTTP.get(url).body, DataFrame)
describe(duncan)

f = Figure()
plt = data(duncan) * mapping(:prestige) * AlgebraOfGraphics.density()
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "prestige_density.svg"), f); # hide

gdf = groupby(duncan, :type)
f = Figure()
plt =
    data(combine(gdf, :prestige => mean)) * mapping(:type, :prestige_mean) * visual(BarPlot)
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "prestige_per_type.svg"), f); # hide

X = Matrix(select(duncan, [:income, :education]))
y = duncan[:, :prestige]
model = robustreg(X, y);

chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain)

quantile(chain)
