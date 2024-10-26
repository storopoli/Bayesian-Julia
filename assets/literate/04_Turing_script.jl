# This file was generated, do not modify it.

using CairoMakie
using Distributions

dice = DiscreteUniform(1, 6)
f, ax, b = barplot(
    dice;
    label="six-sided Dice",
    axis=(; xlabel=L"\theta", ylabel="Mass", xticks=1:6, limits=(nothing, nothing, 0, 0.3)),
)
vlines!(ax, [mean(dice)]; linewidth=5, color=:red, label=L"E(\theta)")
axislegend(ax)
save(joinpath(@OUTPUT, "dice.svg"), f); # hide

using Turing
setprogress!(false) # hide

@model function dice_throw(y)
    #Our prior belief about the probability of each result in a six-sided dice.
    #p is a vector of length 6 each with probability p that sums up to 1.
    p ~ Dirichlet(6, 1)

    #Each outcome of the six-sided dice has a probability p.
    for i in eachindex(y)
        y[i] ~ Categorical(p)
    end
end;

mean(Dirichlet(6, 1))

sum(mean(Dirichlet(6, 1)))

using Random

Random.seed!(123);

my_data = rand(DiscreteUniform(1, 6), 1_000);

first(my_data, 5)

model = dice_throw(my_data);

chain = sample(model, NUTS(), 1_000);

summaries, quantiles = describe(chain);

summaries

sum(summaries[:, :mean])

summarystats(chain[:, 1:3, :])

summarystats(chain[[:var"p[1]", :var"p[2]"]])

sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])

typeof(chain)

using AlgebraOfGraphics
using AlgebraOfGraphics: density
#exclude additional information such as log probability
params = names(chain, :parameters)
chain_mapping =
    mapping(params .=> "sample value") *
    mapping(; color=:chain => nonnumeric, row=dims(1) => renamer(params))
plt1 = data(chain) * mapping(:iteration) * chain_mapping * visual(Lines)
plt2 = data(chain) * chain_mapping * density()
f = Figure(; resolution=(800, 600))
draw!(f[1, 1], plt1)
draw!(f[1, 2], plt2; axis=(; ylabel="density"))
save(joinpath(@OUTPUT, "chain.svg"), f); # hide

prior_chain = sample(model, Prior(), 2_000);

missing_data = similar(my_data, Missing) # vector of `missing`
model_missing = dice_throw(missing_data) # instantiate the "predictive model
prior_check = predict(model_missing, prior_chain);

typeof(prior_check)

summarystats(prior_check[:, 1:5, :]) # just the first 5 prior samples

posterior_check = predict(model_missing, chain);
summarystats(posterior_check[:, 1:5, :]) # just the first 5 posterior samples
