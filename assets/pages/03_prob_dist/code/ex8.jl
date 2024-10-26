# This file was generated, do not modify it. # hide
f, ax, l = lines(
    Exponential(1);
    label=L"\lambda=1",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(0, 4.5, nothing, nothing)),
)
lines!(ax, Exponential(0.5); label=L"\lambda=0.5", linewidth=5)
lines!(ax, Exponential(1.5); label=L"\lambda=2", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "exponential.svg"), f); # hide