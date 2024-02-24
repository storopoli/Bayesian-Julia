# This file was generated, do not modify it. # hide
f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
record(f, joinpath(@OUTPUT, "gibbs_anim.gif"); framerate=5) do frame
    for i in 1:200
        scatter!(ax, (X_gibbs[i, 1], X_gibbs[i, 2]); color=(:red, 0.5))
        linesegments!(X_gibbs[i:(i + 1), 1], X_gibbs[i:(i + 1), 2]; color=(:green, 0.5))
        recordframe!(frame)
    end
end;