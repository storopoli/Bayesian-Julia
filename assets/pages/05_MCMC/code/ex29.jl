# This file was generated, do not modify it. # hide
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