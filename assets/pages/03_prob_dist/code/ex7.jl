# This file was generated, do not modify it. # hide
f, ax, l = lines(
    LogNormal(0, 1);
    label=L"\sigma=1",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(0, 3, nothing, nothing)),
)
lines!(ax, LogNormal(0, 0.25); label=L"\sigma=0.25", linewidth=5)
lines!(ax, LogNormal(0, 0.5); label=L"\sigma=0.5", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "lognormal.svg"), f); # hide