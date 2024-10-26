# This file was generated, do not modify it. # hide
f, ax, l = lines(-4 .. 4, TDist(2); linewidth=5, axis=(xlabel=L"x", ylabel="Density"))
save(joinpath(@OUTPUT, "tdist.svg"), f); # hide