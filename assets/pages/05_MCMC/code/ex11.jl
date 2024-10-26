# This file was generated, do not modify it. # hide
using LinearAlgebra: eigvals, eigvecs
#source: https://discourse.julialang.org/t/plot-ellipse-in-makie/82814/4
function getellipsepoints(cx, cy, rx, ry, θ)
    t = range(0, 2 * pi; length=100)
    ellipse_x_r = @. rx * cos(t)
    ellipse_y_r = @. ry * sin(t)
    R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
    r_ellipse = [ellipse_x_r ellipse_y_r] * R
    x = @. cx + r_ellipse[:, 1]
    y = @. cy + r_ellipse[:, 2]
    return (x, y)
end
function getellipsepoints(μ, Σ; confidence=0.95)
    quant = sqrt(quantile(Chisq(2), confidence))
    cx = μ[1]
    cy = μ[2]

    egvs = eigvals(Σ)
    if egvs[1] > egvs[2]
        idxmax = 1
        largestegv = egvs[1]
        smallesttegv = egvs[2]
    else
        idxmax = 2
        largestegv = egvs[2]
        smallesttegv = egvs[1]
    end

    rx = quant * sqrt(largestegv)
    ry = quant * sqrt(smallesttegv)

    eigvecmax = eigvecs(Σ)[:, idxmax]
    θ = atan(eigvecmax[2] / eigvecmax[1])
    if θ < 0
        θ += 2 * π
    end

    return getellipsepoints(cx, cy, rx, ry, θ)
end

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
record(f, joinpath(@OUTPUT, "met_anim.gif"); framerate=5) do frame
    for i in 1:100
        scatter!(ax, (X_met[i, 1], X_met[i, 2]); color=(:red, 0.5))
        linesegments!(X_met[i:(i + 1), 1], X_met[i:(i + 1), 2]; color=(:green, 0.5))
        recordframe!(frame)
    end
end;