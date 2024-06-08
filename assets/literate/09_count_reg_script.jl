# This file was generated, do not modify it.

using CairoMakie

f, ax, l = lines(-6 .. 6, exp; axis=(xlabel=L"x", ylabel=L"e^x"))
save(joinpath(@OUTPUT, "exponential.svg"), f); # hide

using Turing
using LazyArrays
using Random: seed!
seed!(123)
setprogress!(false) # hide

@model function poissonreg(X, y; predictors=size(X, 2))
    #priors
    α ~ Normal(0, 2.5)
    β ~ filldist(TDist(3), predictors)

    #likelihood
    return y ~ arraydist(LazyArray(@~ LogPoisson.(α .+ X * β)))
end;

using Distributions
f, ax, l = lines(
    Gamma(0.01, 0.01);
    linewidth=2,
    axis=(xlabel=L"\phi", ylabel="Density", limits=(0, 0.03, nothing, nothing)),
)
save(joinpath(@OUTPUT, "gamma.svg"), f); # hide

function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    p = p > 0 ? p : 1e-4 # numerical stability
    r = ϕ

    return NegativeBinomial(r, p)
end

@model function negbinreg(X, y; predictors=size(X, 2))
    #priors
    α ~ Normal(0, 2.5)
    β ~ filldist(TDist(3), predictors)
    ϕ⁻ ~ Gamma(0.01, 0.01)
    ϕ = 1 / ϕ⁻

    #likelihood
    return y ~ arraydist(LazyArray(@~ NegativeBinomial2.(exp.(α .+ X * β), ϕ)))
end;

using DataFrames
using CSV
using HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/roaches.csv"
roaches = CSV.read(HTTP.get(url).body, DataFrame)
describe(roaches)

X = Matrix(select(roaches, Not(:y)))
y = roaches[:, :y]
model_poisson = poissonreg(X, y);

chain_poisson = sample(model_poisson, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain_poisson)

using Chain

@chain quantile(chain_poisson) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(exp); renamecols=false)
end

model_negbin = negbinreg(X, y);

seed!(111) # hide
chain_negbin = sample(model_negbin, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain_negbin)

@chain quantile(chain_negbin) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(exp); renamecols=false)
end
