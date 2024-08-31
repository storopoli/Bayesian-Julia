# This file was generated, do not modify it. # hide
using CairoMakie
using Distributions

f, ax, b = barplot(
    DiscreteUniform(1, 6);
    axis=(;
        title="6-sided Dice",
        xlabel=L"\theta",
        ylabel="Mass",
        xticks=1:6,
        limits=(nothing, nothing, 0, 0.3),
    ),
)
save(joinpath(@OUTPUT, "discrete_uniform.svg"), f); # hide