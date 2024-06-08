# This file was generated, do not modify it. # hide
using AlgebraOfGraphics
params = names(chain, :parameters)
chain_mapping =
    mapping(params .=> "sample value") *
    mapping(; color=:chain => nonnumeric, row=dims(1) => renamer(params))
plt = data(chain) * mapping(:iteration) * chain_mapping * visual(Lines)
f = Figure(; resolution=(1200, 900))
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "traceplot_chain.svg"), f); # hide