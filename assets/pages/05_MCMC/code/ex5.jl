# This file was generated, do not modify it. # hide
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