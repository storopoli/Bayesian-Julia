# This file was generated, do not modify it. # hide
using CairoMakie
using Distributions

f, ax, l = lines(-4 .. 4, Normal(0, 1); linewidth=5, axis=(; xlabel=L"x", ylabel="Density"))
save(joinpath(@OUTPUT, "normal.svg"), f); # hide