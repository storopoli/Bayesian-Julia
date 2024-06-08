# This file was generated, do not modify it. # hide
f = Figure()
plt = data(duncan) * mapping(:prestige) * AlgebraOfGraphics.density()
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "prestige_density.svg"), f); # hide