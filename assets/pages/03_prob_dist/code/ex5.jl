# This file was generated, do not modify it. # hide
f, ax1, b = barplot(
    NegativeBinomial(1, 0.5); axis=(; title=L"k=1", xlabel=L"\theta", ylabel="Mass")
)
ax2 = Axis(f[1, 2]; title=L"k=2", xlabel=L"\theta")
barplot!(ax2, NegativeBinomial(2, 0.5))
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "negbinomial.svg"), f); # hide