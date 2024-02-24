# This file was generated, do not modify it. # hide
using Distributions
f, ax, l = lines(
    Gamma(0.01, 0.01);
    linewidth=2,
    axis=(xlabel=L"\phi", ylabel="Density", limits=(0, 0.03, nothing, nothing)),
)
save(joinpath(@OUTPUT, "gamma.svg"), f); # hide