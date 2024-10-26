# This file was generated, do not modify it. # hide
x = -3:0.01:3
y = -3:0.01:3
dens_mvnormal = [pdf(mvnormal, [i, j]) for i in x, j in y]
f, ax, c = contourf(x, y, dens_mvnormal; axis=(; xlabel=L"X", ylabel=L"Y"))
Colorbar(f[1, 2], c)
save(joinpath(@OUTPUT, "countour_mvnormal.svg"), f); # hide