# This file was generated, do not modify it. # hide
model_intercept_slope = varying_intercept_slope(X, idx, y)
chain_intercept_slope = sample(model_intercept_slope, NUTS(), MCMCThreads(), 1_000, 4)
println(DataFrame(summarystats(chain_intercept_slope)))