# This file was generated, do not modify it. # hide
f, ax1, b = barplot(
    Binomial(5, 0.5); axis=(; title=L"p=0.5", xlabel=L"\theta", ylabel="Mass")
)
ax2 = Axis(f[1, 2]; title=L"p=0.2", xlabel=L"\theta")
barplot!(ax2, Binomial(5, 0.2))
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "binomial.svg"), f); # hide