# This file was generated, do not modify it.

using CairoMakie
using Distributions
using Random

Random.seed!(123);

const N = 100_000
const μ = [0, 0]
const Σ = [1 0.8; 0.8 1]

const mvnormal = MvNormal(μ, Σ)

x = -3:0.01:3
y = -3:0.01:3
dens_mvnormal = [pdf(mvnormal, [i, j]) for i in x, j in y]
f, ax, c = contourf(x, y, dens_mvnormal; axis=(; xlabel=L"X", ylabel=L"Y"))
Colorbar(f[1, 2], c)
save(joinpath(@OUTPUT, "countour_mvnormal.svg"), f); # hide

f, ax, s = surface(
    x,
    y,
    dens_mvnormal;
    axis=(type=Axis3, xlabel=L"X", ylabel=L"Y", zlabel="PDF", azimuth=pi / 8),
)
save(joinpath(@OUTPUT, "surface_mvnormal.svg"), f); # hide

function metropolis(
    S::Int64,
    width::Float64,
    ρ::Float64;
    μ_x::Float64=0.0,
    μ_y::Float64=0.0,
    σ_x::Float64=1.0,
    σ_y::Float64=1.0,
    start_x=-2.5,
    start_y=2.5,
    seed=123,
)
    rgn = MersenneTwister(seed)
    binormal = MvNormal([μ_x; μ_y], [σ_x ρ; ρ σ_y])
    draws = Matrix{Float64}(undef, S, 2)
    accepted = 0::Int64
    x = start_x
    y = start_y
    @inbounds draws[1, :] = [x y]
    for s in 2:S
        x_ = rand(rgn, Uniform(x - width, x + width))
        y_ = rand(rgn, Uniform(y - width, y + width))
        r = exp(logpdf(binormal, [x_, y_]) - logpdf(binormal, [x, y]))

        if r > rand(rgn, Uniform())
            x = x_
            y = y_
            accepted += 1
        end
        @inbounds draws[s, :] = [x y]
    end
    println("Acceptance rate is: $(accepted / S)")
    return draws
end

const S = 10_000
const width = 2.75
const ρ = 0.8

X_met = metropolis(S, width, ρ);

X_met[1:10, :]

using MCMCChains

chain_met = Chains(X_met, [:X, :Y]);

summarystats(chain_met)

mean(summarystats(chain_met)[:, :ess_tail]) / S

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

const warmup = 1_000

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
scatter!(
    ax,
    X_met[warmup:(warmup + 1_000), 1],
    X_met[warmup:(warmup + 1_000), 2];
    color=(:red, 0.3),
)
save(joinpath(@OUTPUT, "met_first1000.svg"), f); # hide

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
scatter!(ax, X_met[warmup:end, 1], X_met[warmup:end, 2]; color=(:red, 0.3))
save(joinpath(@OUTPUT, "met_all.svg"), f); # hide

function gibbs(
    S::Int64,
    ρ::Float64;
    μ_x::Float64=0.0,
    μ_y::Float64=0.0,
    σ_x::Float64=1.0,
    σ_y::Float64=1.0,
    start_x=-2.5,
    start_y=2.5,
    seed=123,
)
    rgn = MersenneTwister(seed)
    binormal = MvNormal([μ_x; μ_y], [σ_x ρ; ρ σ_y])
    draws = Matrix{Float64}(undef, S, 2)
    x = start_x
    y = start_y
    β = ρ * σ_y / σ_x
    λ = ρ * σ_x / σ_y
    sqrt1mrho2 = sqrt(1 - ρ^2)
    σ_YX = σ_y * sqrt1mrho2
    σ_XY = σ_x * sqrt1mrho2
    @inbounds draws[1, :] = [x y]
    for s in 2:S
        if s % 2 == 0
            y = rand(rgn, Normal(μ_y + β * (x - μ_x), σ_YX))
        else
            x = rand(rgn, Normal(μ_x + λ * (y - μ_y), σ_XY))
        end
        @inbounds draws[s, :] = [x y]
    end
    return draws
end

X_gibbs = gibbs(S * 2, ρ);

X_gibbs[1:10, :]

chain_gibbs = Chains(X_gibbs, [:X, :Y]);

summarystats(chain_gibbs)

(mean(summarystats(chain_gibbs)[:, :ess_tail]) / 2) / S

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
record(f, joinpath(@OUTPUT, "gibbs_anim.gif"); framerate=5) do frame
    for i in 1:200
        scatter!(ax, (X_gibbs[i, 1], X_gibbs[i, 2]); color=(:red, 0.5))
        linesegments!(X_gibbs[i:(i + 1), 1], X_gibbs[i:(i + 1), 2]; color=(:green, 0.5))
        recordframe!(frame)
    end
end;

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
scatter!(
    ax,
    X_gibbs[(2 * warmup):(2 * warmup + 1_000), 1],
    X_gibbs[(2 * warmup):(2 * warmup + 1_000), 2];
    color=(:red, 0.3),
)
save(joinpath(@OUTPUT, "gibbs_first1000.svg"), f); # hide

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
scatter!(ax, X_gibbs[(2 * warmup):end, 1], X_gibbs[(2 * warmup):end, 2]; color=(:red, 0.3))
save(joinpath(@OUTPUT, "gibbs_all.svg"), f); # hide

const starts = collect(Iterators.product((-2.5, 2.5), (2.5, -2.5)))

const S_parallel = 100;

X_met_1 = metropolis(
    S_parallel, width, ρ; seed=124, start_x=first(starts[1]), start_y=last(starts[1])
);
X_met_2 = metropolis(
    S_parallel, width, ρ; seed=125, start_x=first(starts[2]), start_y=last(starts[2])
);
X_met_3 = metropolis(
    S_parallel, width, ρ; seed=126, start_x=first(starts[3]), start_y=last(starts[3])
);
X_met_4 = metropolis(
    S_parallel, width, ρ; seed=127, start_x=first(starts[4]), start_y=last(starts[4])
);

using Colors
const logocolors = Colors.JULIA_LOGO_COLORS;
f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
record(f, joinpath(@OUTPUT, "parallel_met.gif"); framerate=5) do frame
    for i in 1:99
        scatter!(ax, (X_met_1[i, 1], X_met_1[i, 2]); color=(logocolors.blue, 0.5))
        linesegments!(
            X_met_1[i:(i + 1), 1], X_met_1[i:(i + 1), 2]; color=(logocolors.blue, 0.5)
        )
        scatter!(ax, (X_met_2[i, 1], X_met_2[i, 2]); color=(logocolors.red, 0.5))
        linesegments!(
            X_met_2[i:(i + 1), 1], X_met_2[i:(i + 1), 2]; color=(logocolors.red, 0.5)
        )
        scatter!(ax, (X_met_3[i, 1], X_met_3[i, 2]); color=(logocolors.green, 0.5))
        linesegments!(
            X_met_3[i:(i + 1), 1], X_met_3[i:(i + 1), 2]; color=(logocolors.green, 0.5)
        )
        scatter!(ax, (X_met_4[i, 1], X_met_4[i, 2]); color=(logocolors.purple, 0.5))
        linesegments!(
            X_met_4[i:(i + 1), 1], X_met_4[i:(i + 1), 2]; color=(logocolors.purple, 0.5)
        )
        recordframe!(frame)
    end
end;

X_gibbs_1 = gibbs(
    S_parallel * 2, ρ; seed=124, start_x=first(starts[1]), start_y=last(starts[1])
);
X_gibbs_2 = gibbs(
    S_parallel * 2, ρ; seed=125, start_x=first(starts[2]), start_y=last(starts[2])
);
X_gibbs_3 = gibbs(
    S_parallel * 2, ρ; seed=126, start_x=first(starts[3]), start_y=last(starts[3])
);
X_gibbs_4 = gibbs(
    S_parallel * 2, ρ; seed=127, start_x=first(starts[4]), start_y=last(starts[4])
);

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
record(f, joinpath(@OUTPUT, "parallel_gibbs.gif"); framerate=5) do frame
    for i in 1:199
        scatter!(ax, (X_gibbs_1[i, 1], X_gibbs_1[i, 2]); color=(logocolors.blue, 0.5))
        linesegments!(
            X_gibbs_1[i:(i + 1), 1], X_gibbs_1[i:(i + 1), 2]; color=(logocolors.blue, 0.5)
        )
        scatter!(ax, (X_gibbs_2[i, 1], X_gibbs_2[i, 2]); color=(logocolors.red, 0.5))
        linesegments!(
            X_gibbs_2[i:(i + 1), 1], X_gibbs_2[i:(i + 1), 2]; color=(logocolors.red, 0.5)
        )
        scatter!(ax, (X_gibbs_3[i, 1], X_gibbs_3[i, 2]); color=(logocolors.green, 0.5))
        linesegments!(
            X_gibbs_3[i:(i + 1), 1], X_gibbs_3[i:(i + 1), 2]; color=(logocolors.green, 0.5)
        )
        scatter!(ax, (X_gibbs_4[i, 1], X_gibbs_4[i, 2]); color=(logocolors.purple, 0.5))
        linesegments!(
            X_gibbs_4[i:(i + 1), 1], X_gibbs_4[i:(i + 1), 2]; color=(logocolors.purple, 0.5)
        )
        recordframe!(frame)
    end
end;

using ForwardDiff: gradient
function hmc(
    S::Int64,
    width::Float64,
    ρ::Float64;
    L=40,
    ϵ=0.001,
    μ_x::Float64=0.0,
    μ_y::Float64=0.0,
    σ_x::Float64=1.0,
    σ_y::Float64=1.0,
    start_x=-2.5,
    start_y=2.5,
    seed=123,
)
    rgn = MersenneTwister(seed)
    binormal = MvNormal([μ_x; μ_y], [σ_x ρ; ρ σ_y])
    draws = Matrix{Float64}(undef, S, 2)
    accepted = 0::Int64
    x = start_x
    y = start_y
    @inbounds draws[1, :] = [x y]
    M = [1.0 0.0; 0.0 1.0]
    ϕ_d = MvNormal([0.0, 0.0], M)
    for s in 2:S
        x_ = rand(rgn, Uniform(x - width, x + width))
        y_ = rand(rgn, Uniform(y - width, y + width))
        ϕ = rand(rgn, ϕ_d)
        kinetic = sum(ϕ .^ 2) / 2
        log_p = logpdf(binormal, [x, y]) - kinetic
        ϕ += 0.5 * ϵ * gradient(x -> logpdf(binormal, x), [x_, y_])
        for l in 1:L
            x_, y_ = [x_, y_] + (ϵ * M * ϕ)
            ϕ += +0.5 * ϵ * gradient(x -> logpdf(binormal, x), [x_, y_])
        end
        ϕ = -ϕ # make the proposal symmetric
        kinetic = sum(ϕ .^ 2) / 2
        log_p_ = logpdf(binormal, [x_, y_]) - kinetic
        r = exp(log_p_ - log_p)

        if r > rand(rgn, Uniform())
            x = x_
            y = y_
            accepted += 1
        end
        @inbounds draws[s, :] = [x y]
    end
    println("Acceptance rate is: $(accepted / S)")
    return draws
end

gradient(x -> logpdf(mvnormal, x), [1, -1])

X_hmc = hmc(S, width, ρ; ϵ=0.0856, L=40);

X_hmc[1:10, :]

chain_hmc = Chains(X_hmc, [:X, :Y]);

summarystats(chain_hmc)

mean(summarystats(chain_hmc)[:, :ess_tail]) / S

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
record(f, joinpath(@OUTPUT, "hmc_anim.gif"); framerate=5) do frame
    for i in 1:100
        scatter!(ax, (X_hmc[i, 1], X_hmc[i, 2]); color=(:red, 0.5))
        linesegments!(X_hmc[i:(i + 1), 1], X_hmc[i:(i + 1), 2]; color=(:green, 0.5))
        recordframe!(frame)
    end
end;

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
scatter!(
    ax,
    X_hmc[warmup:(warmup + 1_000), 1],
    X_hmc[warmup:(warmup + 1_000), 2];
    color=(:red, 0.3),
)
save(joinpath(@OUTPUT, "hmc_first1000.svg"), f); # hide

f, ax, l = lines(
    getellipsepoints(μ, Σ; confidence=0.9)...;
    label="90% HPD",
    linewidth=2,
    axis=(; limits=(-3, 3, -3, 3), xlabel=L"\theta_1", ylabel=L"\theta_2"),
)
axislegend(ax)
scatter!(ax, X_hmc[warmup:end, 1], X_hmc[warmup:end, 2]; color=(:red, 0.3))
save(joinpath(@OUTPUT, "hmc_all.svg"), f); # hide

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

funnel_y = rand(Normal(0, 3), 10_000)
funnel_x = rand(Normal(), 10_000) .* exp.(funnel_y / 2)

f, ax, s = scatter(
    funnel_x,
    funnel_y;
    color=(:steelblue, 0.3),
    axis=(; xlabel=L"X", ylabel=L"Y", limits=(-100, 100, nothing, nothing)),
)
save(joinpath(@OUTPUT, "funnel.svg"), f); # hide

using Turing
setprogress!(false) # hide

@model function dice_throw(y)
    #Our prior belief about the probability of each result in a six-sided dice.
    #p is a vector of length 6 each with probability p that sums up to 1.
    p ~ Dirichlet(6, 1)

    #Each outcome of the six-sided dice has a probability p.
    return y ~ filldist(Categorical(p), length(y))
end;

data_dice = rand(DiscreteUniform(1, 6), 1_000);

model = dice_throw(data_dice)
chain = sample(model, NUTS(), 1_000);
summarystats(chain)

bad_chain = sample(model, NUTS(0.3), 500)
summarystats(bad_chain)

sum(bad_chain[:numerical_error])

mean(bad_chain[:acceptance_rate])

mean(chain[:acceptance_rate])

using AlgebraOfGraphics
params = names(chain, :parameters)
chain_mapping =
    mapping(params .=> "sample value") *
    mapping(; color=:chain => nonnumeric, row=dims(1) => renamer(params))
plt = data(chain) * mapping(:iteration) * chain_mapping * visual(Lines)
f = Figure(; resolution=(1200, 900))
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "traceplot_chain.svg"), f); # hide

params = names(bad_chain, :parameters)
chain_mapping =
    mapping(params .=> "sample value") *
    mapping(; color=:chain => nonnumeric, row=dims(1) => renamer(params))
plt = data(bad_chain) * mapping(:iteration) * chain_mapping * visual(Lines)
f = Figure(; resolution=(1200, 900))
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "traceplot_bad_chain.svg"), f); # hide
