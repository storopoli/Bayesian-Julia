# This file was generated, do not modify it. # hide
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