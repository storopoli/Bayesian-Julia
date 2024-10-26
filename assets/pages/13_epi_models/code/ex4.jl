# This file was generated, do not modify it. # hide
using AlgebraOfGraphics
using CairoMakie
f = Figure()
plt = data(br) * mapping(:date => L"t", :new_confirmed => "infected daily") * visual(Lines)
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "infected.svg"), f); # hide