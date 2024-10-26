# This file was generated, do not modify it. # hide
X_correlated = hcat(fill(1, size(X, 1)), X)
model_correlated = correlated_varying_intercept_slope(X_correlated, idx, y)
chain_correlated = sample(model_correlated, NUTS(), MCMCThreads(), 1_000, 4)
println(DataFrame(summarystats(chain_correlated)))