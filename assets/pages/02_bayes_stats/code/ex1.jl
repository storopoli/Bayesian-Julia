# This file was generated, do not modify it. # hide
using CairoMakie
using Distributions

d = LogNormal(0, 2)
range_d = 0:0.001:4
q25 = quantile(d, 0.25)
q75 = quantile(d, 0.75)
credint = range(q25; stop=q75, length=100)
f, ax, l = lines(
    range_d,
    pdf.(d, range_d);
    linewidth=3,
    axis=(; limits=(-0.2, 4.2, nothing, nothing), xlabel=L"\theta", ylabel="Density"),
)
scatter!(ax, mode(d), pdf(d, mode(d)); color=:green, markersize=12)
band!(ax, credint, 0.0, pdf.(d, credint); color=(:steelblue, 0.5))
save(joinpath(@OUTPUT, "lognormal.svg"), f); # hide