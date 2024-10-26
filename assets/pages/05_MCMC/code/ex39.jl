# This file was generated, do not modify it. # hide
d1 = MvNormal([10, 2], [1 0; 0 1])
d2 = MvNormal([0, 0], [8.4 2.0; 2.0 1.7])

d = MixtureModel([d1, d2])

x = -6:0.01:15
y = -2.5:0.01:4.2
dens_mixture = [pdf(d, [i, j]) for i in x, j in y]

f, ax, s = surface(
    x,
    y,
    dens_mixture;
    axis=(type=Axis3, xlabel=L"X", ylabel=L"Y", zlabel="PDF", azimuth=pi / 4),
)
save(joinpath(@OUTPUT, "bimodal.svg"), f); # hide