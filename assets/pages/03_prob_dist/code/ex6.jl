# This file was generated, do not modify it. # hide
f, ax, l = lines(
    Normal(0, 1);
    label=L"\sigma=1",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(-4, 4, nothing, nothing)),
)
lines!(ax, Normal(0, 0.5); label=L"\sigma=0.5", linewidth=5)
lines!(ax, Normal(0, 2); label=L"\sigma=2", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "normal.svg"), f); # hide