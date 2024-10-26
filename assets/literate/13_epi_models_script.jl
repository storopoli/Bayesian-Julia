# This file was generated, do not modify it.

using Downloads
using DataFrames
using CSV
using Chain
using Dates

url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
file = Downloads.download(url)
df = DataFrame(CSV.File(file))
br = @chain df begin
    filter(
        [:date, :city] =>
            (date, city) ->
                date < Dates.Date("2021-01-01") &&
                    date > Dates.Date("2020-04-01") &&
                    ismissing(city),
        _,
    )
    groupby(:date)
    combine(
        [
            :estimated_population_2019,
            :last_available_confirmed_per_100k_inhabitants,
            :last_available_deaths,
            :new_confirmed,
            :new_deaths,
        ] .=>
            sum .=> [
                :estimated_population_2019,
                :last_available_confirmed_per_100k_inhabitants,
                :last_available_deaths,
                :new_confirmed,
                :new_deaths,
            ],
    )
end;

first(br, 5)

last(br, 5)

using AlgebraOfGraphics
using CairoMakie
f = Figure()
plt = data(br) * mapping(:date => L"t", :new_confirmed => "infected daily") * visual(Lines)
draw!(f[1, 1], plt)
save(joinpath(@OUTPUT, "infected.svg"), f); # hide

using DifferentialEquations

function sir_ode!(du, u, p, t)
    (S, I, R) = u
    (β, γ) = p
    N = S + I + R
    infection = β * I * S / N
    recovery = γ * I
    @inbounds begin
        du[1] = -infection # Susceptible
        du[2] = infection - recovery # Infected
        du[3] = recovery # Recovered
    end
    return nothing
end;

i₀ = first(br[:, :new_confirmed])
N = maximum(br[:, :estimated_population_2019])

u = [N - i₀, i₀, 0.0]
p = [0.5, 0.05]
prob = ODEProblem(sir_ode!, u, (1.0, 100.0), p)
sol_ode = solve(prob)
f = Figure()
plt =
    data(stack(DataFrame(sol_ode), Not(:timestamp))) *
    mapping(
        :timestamp => L"t",
        :value => L"N";
        color=:variable => renamer(["value1" => "S", "value2" => "I", "value3" => "R"]),
    ) *
    visual(Lines; linewidth=3)
draw!(f[1, 1], plt; axis=(; title="SIR Model for 100 days, β = $(p[1]), γ = $(p[2])"))
save(joinpath(@OUTPUT, "ode_solve.svg"), f); # hide

function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ

    return NegativeBinomial(r, p)
end

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

infected = br[:, :new_confirmed]
r₀ = first(br[:, :new_deaths])
model_sir = bayes_sir(infected, i₀, r₀, N)
chain_sir = sample(model_sir, NUTS(), 1_000)
summarystats(chain_sir[[:β, :γ]])
