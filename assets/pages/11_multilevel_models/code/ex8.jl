# This file was generated, do not modify it. # hide
model_intercept = varying_intercept(X, idx, y)
chain_intercept = sample(model_intercept, NUTS(), MCMCThreads(), 1_000, 4)
println(DataFrame(summarystats(chain_intercept)))