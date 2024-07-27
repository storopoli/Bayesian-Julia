# This file was generated, do not modify it. # hide
f, ax, l = lines(
    -4 .. 4,
    Normal(0, 1);
    linewidth=5,
    label="Normal",
    axis=(; xlabel=L"x", ylabel="Density"),
)
lines!(ax, -4 .. 4, TDist(2); linewidth=5, label="Student")
axislegend(ax)
save(joinpath(@OUTPUT, "comparison_normal_student.svg"), f); # hide