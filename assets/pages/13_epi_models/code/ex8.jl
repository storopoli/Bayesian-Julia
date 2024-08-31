# This file was generated, do not modify it. # hide
using Turing
using LazyArrays
using Random: seed!
seed!(123)
setprogress!(false) # hide

@model function bayes_sir(infected, i₀, r₀, N)
    #calculate number of timepoints
    l = length(infected)

    #priors
    β ~ TruncatedNormal(2, 1, 1e-4, 10)     # using 10 because numerical issues arose
    γ ~ TruncatedNormal(0.4, 0.5, 1e-4, 10) # using 10 because numerical issues arose
    ϕ⁻ ~ truncated(Exponential(5); lower=0, upper=1e5)
    ϕ = 1.0 / ϕ⁻

    #ODE Stuff
    I = i₀
    u0 = [N - I, I, r₀] # S,I,R
    p = [β, γ]
    tspan = (1.0, float(l))
    prob = ODEProblem(sir_ode!, u0, tspan, p)
    sol = solve(
        prob,
        Tsit5(); # similar to Dormand-Prince RK45 in Stan but 20% faster
        saveat=1.0,
    )
    solᵢ = Array(sol)[2, :] # New Infected
    solᵢ = max.(1e-4, solᵢ) # numerical issues arose

    #likelihood
    return infected ~ arraydist(LazyArray(@~ NegativeBinomial2.(solᵢ, ϕ)))
end;