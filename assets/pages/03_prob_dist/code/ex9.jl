# This file was generated, do not modify it. # hide
f, ax, l = lines(
    TDist(2);
    label=L"\nu=2",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(-4, 4, nothing, nothing)),
)
lines!(ax, TDist(8); label=L"\nu=8", linewidth=5)
lines!(ax, TDist(30); label=L"\nu=30", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "tdist.svg"), f); # hide