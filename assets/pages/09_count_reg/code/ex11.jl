# This file was generated, do not modify it. # hide
seed!(111) # hide
chain_negbin = sample(model_negbin, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain_negbin)