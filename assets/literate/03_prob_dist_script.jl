# This file was generated, do not modify it.

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

f, ax1, b = barplot(
    Bernoulli(0.5);
    width=0.3,
    axis=(;
        title=L"p=0.5",
        xlabel=L"\theta",
        ylabel="Mass",
        xticks=0:1,
        limits=(nothing, nothing, 0, 1),
    ),
)
ax2 = Axis(
    f[1, 2]; title=L"p=0.2", xlabel=L"\theta", xticks=0:1, limits=(nothing, nothing, 0, 1)
)
barplot!(ax2, Bernoulli(0.2); width=0.3)
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "bernoulli.svg"), f); # hide

f, ax1, b = barplot(
    Binomial(5, 0.5); axis=(; title=L"p=0.5", xlabel=L"\theta", ylabel="Mass")
)
ax2 = Axis(f[1, 2]; title=L"p=0.2", xlabel=L"\theta")
barplot!(ax2, Binomial(5, 0.2))
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "binomial.svg"), f); # hide

f, ax1, b = barplot(
    Poisson(1); axis=(; title=L"\lambda=1", xlabel=L"\theta", ylabel="Mass")
)
ax2 = Axis(f[1, 2]; title=L"\lambda=4", xlabel=L"\theta")
barplot!(ax2, Poisson(4))
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "poisson.svg"), f); # hide

f, ax1, b = barplot(
    NegativeBinomial(1, 0.5); axis=(; title=L"k=1", xlabel=L"\theta", ylabel="Mass")
)
ax2 = Axis(f[1, 2]; title=L"k=2", xlabel=L"\theta")
barplot!(ax2, NegativeBinomial(2, 0.5))
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "negbinomial.svg"), f); # hide

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
