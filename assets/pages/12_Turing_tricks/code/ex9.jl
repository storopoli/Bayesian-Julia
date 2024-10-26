# This file was generated, do not modify it. # hide
using CairoMakie
using Distributions
funnel_y = rand(Normal(0, 3), 10_000)
funnel_x = rand(Normal(), 10_000) .* exp.(funnel_y / 2)

f, ax, s = scatter(
    funnel_x,
    funnel_y;
    color=(:steelblue, 0.3),
    axis=(; xlabel=L"X", ylabel=L"Y", limits=(-100, 100, nothing, nothing)),
)
save(joinpath(@OUTPUT, "funnel.svg"), f); # hide