# This file was generated, do not modify it. # hide
f, ax1, b = barplot(
    Poisson(1); axis=(; title=L"\lambda=1", xlabel=L"\theta", ylabel="Mass")
)
ax2 = Axis(f[1, 2]; title=L"\lambda=4", xlabel=L"\theta")
barplot!(ax2, Poisson(4))
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "poisson.svg"), f); # hide