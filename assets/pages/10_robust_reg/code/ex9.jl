# This file was generated, do not modify it. # hide
chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain)