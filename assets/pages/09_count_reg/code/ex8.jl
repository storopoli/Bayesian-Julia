# This file was generated, do not modify it. # hide
chain_poisson = sample(model_poisson, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain_poisson)