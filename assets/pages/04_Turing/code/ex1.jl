# This file was generated, do not modify it. # hide
using CairoMakie
using Distributions

dice = DiscreteUniform(1, 6)
f, ax, b = barplot(
    dice;
    label="six-sided Dice",
    axis=(; xlabel=L"\theta", ylabel="Mass", xticks=1:6, limits=(nothing, nothing, 0, 0.3)),
)
vlines!(ax, [mean(dice)]; linewidth=5, color=:red, label=L"E(\theta)")
axislegend(ax)
save(joinpath(@OUTPUT, "dice.svg"), f); # hide