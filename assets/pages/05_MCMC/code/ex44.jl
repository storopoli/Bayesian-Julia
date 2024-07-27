# This file was generated, do not modify it. # hide
bad_chain = sample(model, NUTS(0.3), 500)
summarystats(bad_chain)