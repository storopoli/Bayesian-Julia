# This file was generated, do not modify it. # hide
gdf = groupby(duncan, :type)
f = Figure()
plt =
    data(combine(gdf, :prestige => mean)) * mapping(:type, :prestige_mean) * visual(BarPlot)
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "prestige_per_type.svg"), f); # hide