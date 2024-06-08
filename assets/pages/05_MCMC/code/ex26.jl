# This file was generated, do not modify it. # hide
using Colors
const logocolors = Colors.JULIA_LOGO_COLORS;
f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
record(f, joinpath(@OUTPUT, "parallel_met.gif"); framerate=5) do frame
    for i in 1:99
        scatter!(ax, (X_met_1[i, 1], X_met_1[i, 2]); color=(logocolors.blue, 0.5))
        linesegments!(
            X_met_1[i:(i + 1), 1], X_met_1[i:(i + 1), 2]; color=(logocolors.blue, 0.5)
        )
        scatter!(ax, (X_met_2[i, 1], X_met_2[i, 2]); color=(logocolors.red, 0.5))
        linesegments!(
            X_met_2[i:(i + 1), 1], X_met_2[i:(i + 1), 2]; color=(logocolors.red, 0.5)
        )
        scatter!(ax, (X_met_3[i, 1], X_met_3[i, 2]); color=(logocolors.green, 0.5))
        linesegments!(
            X_met_3[i:(i + 1), 1], X_met_3[i:(i + 1), 2]; color=(logocolors.green, 0.5)
        )
        scatter!(ax, (X_met_4[i, 1], X_met_4[i, 2]); color=(logocolors.purple, 0.5))
        linesegments!(
            X_met_4[i:(i + 1), 1], X_met_4[i:(i + 1), 2]; color=(logocolors.purple, 0.5)
        )
        recordframe!(frame)
    end
end;