# This file was generated, do not modify it. # hide
using AlgebraOfGraphics
using AlgebraOfGraphics: density
#exclude additional information such as log probability
params = names(chain, :parameters)
chain_mapping =
    mapping(params .=> "sample value") *
    mapping(; color=:chain => nonnumeric, row=dims(1) => renamer(params))
plt1 = data(chain) * mapping(:iteration) * chain_mapping * visual(Lines)
plt2 = data(chain) * chain_mapping * density()
f = Figure(; resolution=(800, 600))
draw!(f[1, 1], plt1)
draw!(f[1, 2], plt2; axis=(; ylabel="density"))
save(joinpath(@OUTPUT, "chain.svg"), f); # hide