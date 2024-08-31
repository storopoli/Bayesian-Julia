# This file was generated, do not modify it. # hide
@model function ncp_funnel()
    x̃ ~ Normal()
    ỹ ~ Normal()
    y = 3.0 * ỹ         # implies y ~ Normal(0, 3)
    return x = exp(y / 2) * x̃  # implies x ~ Normal(0, exp(y / 2))
end

chain_ncp_funnel = sample(ncp_funnel(), NUTS(), MCMCThreads(), 1_000, 4)