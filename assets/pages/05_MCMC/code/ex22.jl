# This file was generated, do not modify it. # hide
f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
scatter!(ax, X_gibbs[(2 * warmup):end, 1], X_gibbs[(2 * warmup):end, 2]; color=(:red, 0.3))
save(joinpath(@OUTPUT, "gibbs_all.svg"), f); # hide