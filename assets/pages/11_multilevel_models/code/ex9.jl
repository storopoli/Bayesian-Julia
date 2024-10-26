# This file was generated, do not modify it. # hide
model_slope = varying_slope(X, idx, y)
chain_slope = sample(model_slope, NUTS(), MCMCThreads(), 1_000, 4)
println(DataFrame(summarystats(chain_slope)))