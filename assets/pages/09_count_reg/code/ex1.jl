# This file was generated, do not modify it. # hide
using CairoMakie

f, ax, l = lines(-6 .. 6, exp; axis=(xlabel=L"x", ylabel=L"e^x"))
save(joinpath(@OUTPUT, "exponential.svg"), f); # hide