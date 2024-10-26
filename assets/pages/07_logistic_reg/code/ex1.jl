# This file was generated, do not modify it. # hide
using CairoMakie

function logistic(x)
    return 1 / (1 + exp(-x))
end

f, ax, l = lines(-10 .. 10, logistic; axis=(; xlabel=L"x", ylabel=L"\mathrm{Logistic}(x)"))
f
save(joinpath(@OUTPUT, "logistic.svg"), f); # hide