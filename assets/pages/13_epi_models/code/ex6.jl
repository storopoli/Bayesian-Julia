# This file was generated, do not modify it. # hide
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