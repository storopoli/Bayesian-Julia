# This file was generated, do not modify it. # hide
f, ax, s = surface(
    x,
    y,
    dens_mvnormal;
    axis=(type=Axis3, xlabel=L"X", ylabel=L"Y", zlabel="PDF", azimuth=pi / 8),
)
save(joinpath(@OUTPUT, "surface_mvnormal.svg"), f); # hide