# This file was generated, do not modify it. # hide
f, ax, l = lines(
    Beta(1, 1);
    label=L"a=b=1",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(0, 1, nothing, nothing)),
)
lines!(ax, Beta(3, 2); label=L"a=3, b=2", linewidth=5)
lines!(ax, Beta(2, 3); label=L"a=2, b=3", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "beta.svg"), f); # hide