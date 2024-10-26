# This file was generated, do not modify it. # hide
infected = br[:, :new_confirmed]
r₀ = first(br[:, :new_deaths])
model_sir = bayes_sir(infected, i₀, r₀, N)
chain_sir = sample(model_sir, NUTS(), 1_000)
summarystats(chain_sir[[:β, :γ]])