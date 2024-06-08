# This file was generated, do not modify it. # hide
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