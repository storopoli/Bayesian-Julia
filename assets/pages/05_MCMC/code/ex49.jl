# This file was generated, do not modify it. # hide
params = names(bad_chain, :parameters)
chain_mapping =
    mapping(params .=> "sample value") *
    mapping(; color=:chain => nonnumeric, row=dims(1) => renamer(params))
plt = data(bad_chain) * mapping(:iteration) * chain_mapping * visual(Lines)
f = Figure(; resolution=(1200, 900))
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "traceplot_bad_chain.svg"), f); # hide