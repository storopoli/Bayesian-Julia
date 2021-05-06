# # Robust Bayesian Regression

# Leaving the universe of linear models, we start to venture into generalized linear models (GLM). The third of these is
# **robust regression**.

# A regression with count data behaves exactly like a linear model: it makes a prediction simply by computing a weighted
# sum of the independent variables $\mathbf{X}$ by the estimated coefficients $\boldsymbol{\beta}$, plus an intercept
# $\alpha$. However, instead of using a Gaussian/normal likelihood function, it uses a **Student-$t$ likelihood function**.

# We use robust regression in the same context as linear regression: our dependent variable is continuous. But robust regression
# allows us to **better handle outliers** in our data.

# Before we dive in the nuts and bolts of robust regression let's remember the Gaussian/normal curve that has a bell shape
# (figure below). It does not have a "fat tail" (or sometimes known as "long tail"). In other words, the observations are
# not far from the mean. When we use this distribution as a likelihood function in the Bayesian models, we force that all
# estimates must be conditioned into a normal distribution of the dependent variable. If there are many outliers in the
# data (observations quite far from the mean), this causes the estimates of of the independent variables' coefficients to
# be unstable. This is because the normal distribution cannot contemplate observations that are very spread away from the
# mean without having to change the mean's position (or location). In other words, the bell curve needs to "shift" to be able
# to contemplate outliers, thus making the inference unstable.

using StatsPlots, Distributions, LaTeXStrings

plot(Normal(0, 1),
        lw=5, label=false,
        xlabel=L"x",
        ylabel="Density")
savefig(joinpath(@OUTPUT, "normal.svg")); # hide

# \fig{normal}
# \center{*Normal with $\mu=0$ and $\sigma = 1$*} \\

# So we need a more "malleable" distribution as a likelihood function. A distribution that is more robust to outliers.
# A distribution similar to Normal but that has "fatter" (or "longer") tails to precisely contemplate observations very
# far from the average without having to "shift" the mean's position (or location). For that we have the Student-$t$ distribution.
# See the figure below to remember its shape.

plot(TDist(2),
        lw=5, label=false,
        xlabel=L"x",
        ylabel="Density",
        xlims=(-4, 4))
savefig(joinpath(@OUTPUT, "tdist.svg")); # hide

# \fig{tdist}
# \center{*Student-$t$ with $\nu =  2$*} \\

# ## Comparison Between Normal vs Student-$t$

# Take a look at the tails in the comparison below:

plot(Normal(0, 1),
        lw=5, label="Normal",
        xlabel=L"x",
        ylabel="Density",
        xlims=(-4, 4))
plot!(TDist(2),
        lw=5, label="Student")
savefig(joinpath(@OUTPUT, "comparison_normal_student.svg")); # hide

# \fig{comparison_normal_student}
# \center{*Comparison between Normal and Student-$t$ Distributions*} \\

# ## Bayesian Robust Regression

# The standard approach for modeling a continuous dependent variable is with a Gaussian/normal likelihood function.
# This implies that the model error, $\sigma$ of the Gaussian/normal likelihood function is distributed as a normal distribution:

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Normal}\left( \alpha + \mathbf{X} \cdot \boldsymbol{\beta}, \sigma \right) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \boldsymbol{\beta} &\sim \text{Normal}(\mu_{\boldsymbol{\beta}}, \sigma_{\boldsymbol{\beta}}) \\
# \sigma &\sim \text{Exponential}(\lambda_\sigma)
# \end{aligned}
# $$

# From a Bayesian point of view, there is nothing special about Gaussian/normal likelihood function
# It is just a probabilistic distribution specified in a model. We can make the model more robust
# by using a Student-$t$ distribution as a likelihood function. This implies that the model error,
# $\sigma$ does not follow a normal distribution, instead it follows a Student-$t$ distribution:

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Student}\left( \nu, \alpha + \mathbf{X} \cdot \boldsymbol{\beta}, \sigma \right) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \boldsymbol{\beta} &\sim \text{Normal}(\mu_{\boldsymbol{\beta}}, \sigma_{\boldsymbol{\beta}}) \\
# \nu &\sim \text{Log-Normal}(2, 1) \\
# \sigma &\sim \text{Exponential}(\lambda_\sigma)
# \end{aligned}
# $$

# Here are some differences:

# 1. Student-$t$ likelihood function requires one additional parameter: $\nu$, degrees of freedom. These control how "fat" (or "long") the tails will be. Values of $\nu> 20$ forces the Student-$t$ distribution to practically become a normal distribution.
# 2. There is nothing special about $\nu$. It is just another parameter for the model to estimate. So just specify a prior on it. In this case, I am using a Log-Normal distribution with mean 2 and standard deviation 1.

# Note that there is also nothing special about the priors of the $\boldsymbol{\beta}$ coefficients or the intercept $\alpha$.
# We could very well also specify other distributions as priors or even make the model even more robust to outliers by
# specifying priors as Student-$t$ distributions:

# $$
# \begin{aligned}
# \alpha &\sim \text{Student}(\nu_\alpha, \mu_\alpha, \sigma_\alpha) \\
# \boldsymbol{\beta} &\sim \text{Student}(\nu_{\boldsymbol{\beta}}, \mu_{\boldsymbol{\beta}}, \sigma_{\boldsymbol{\beta}}) \\
# \nu_\alpha &\sim \text{Log-Normal}(1, 1) \\
# \nu_{\boldsymbol{\beta}} &\sim \text{Log-Normal}(1, 1)
# \end{aligned}
# $$

# Our goal is to instantiate a regression with count data using the observed data ($\mathbf{y}$ and $\mathbf{X}$) and find the posterior
# distribution of our model's parameters of interest ($\alpha$ and $\boldsymbol{\beta}$). This means to find the full posterior
# distribution of:

# $$ P(\boldsymbol{\theta} \mid \mathbf{y}) = P(\alpha, \boldsymbol{\beta}, \sigma \mid \mathbf{y}) $$

# This is easily accomplished with Turing:

using Turing
using Statistics: mean, std
using StatsBase:mad
using Random:seed!
seed!(123)
setprogress!(false) # hide

@model robustreg(X, y; predictors=size(X, 2)) = begin
    #priors
    νₐ ~ LogNormal(1, 1)
    νᵦ ~ LogNormal(1, 1)
    α ~ LocationScale(median(y), 2.5 * mad(y), TDist(νₐ))
    β ~ filldist(TDist(νᵦ), predictors)
    σ ~ Exponential(1)
    ν ~ LogNormal(2, 1)

    #likelihood
    y .~ LocationScale.(α .+ X * β, σ, TDist(ν))
end;

# Here I am specifying very weakly informative priors:

# * $\alpha \sim \text{Student-}t(\operatorname{median}(\mathbf{y}), 2.5 \cdot \operatorname{MAD}(\mathbf{y}), \nu_{\alpha})$ -- This means a Student-$t$ distribution centered on `y`'s median with variance 2.5 times the mean absolute deviation (MAD) of `y`. That prior should with ease cover all possible values of $\alpha$. Remember that the Student-$t$ distribution has support over all the real number line $\in (-\infty, +\infty)$. The `LocationScale()` Turing's function adds location and scale parameters to distributions that doesn't have it. This is the case with the `TDist()` distribution which only takes the `ν` degrees of of freedom as parameter.
# * $\boldsymbol{\beta} \sim \text{Student-}t(0,1,\nu_{\boldsymbol{\beta}})$ -- The predictors all have a prior distribution of a Student-$t$ distribution centered on 0 with variance 1 and degrees of freedom $\nu_{\boldsymbol{\beta}}$. That wide-tailed $t$ distribution will cover all possible values for our coefficients. Remember the Student-$t$ also has support over all the real number line $\in (-\infty, +\infty)$. Also the `filldist()` is a nice Turing's function which takes any univariate or multivariate distribution and returns another distribution that repeats the input distribution.
# * $\sigma \sim \text{Exponential}(1)$ -- A wide-tailed-positive-only distribution perfectly suited for our model's error.

# Finally, the likelihood function is familiar dot `.` broadcasting operator in Julia. By specifying that `y` vector is
# "broadcasted distributed" `.~` as a `LocationScale` broadcasted to mean (location parameter) `α` added to the product
# of the data matrix `X` and `β` coefficient vector along with a variance (scale parameter) `σ`. To conclude, we place
# inside the `LocationScale` a broadcasted `TDist` with `ν` degrees of freedom parameter.

# ## Example - Duncan's Prestige

# For our example, I will use a famous dataset called `duncan` (Duncan, 1961), which is data from occupation's prestige filled with
# outliers.
# It has 45 observations and the following variables:

# * `profession`: name of the profession.
# * `type`: type of occupation. A qualitative variable:
#      * `prof` - professional or management.
#      * `wc` - white-collar.
#      * `bc` - blue-collar.
# * `income`: percentage of people in the occupation earning over U\$ 3,500 per year in 1950 (more or less U\$ 36,000 in 2017).
# * `education`: percentage of people in the occupation who had a high school diploma in 1949 (which, being cynical, we can say is somewhat equivalent to a PhD degree in 2017).
# * `prestige`: percentage of respondents in the survey who classified their occupation as at least "good" with respect to prestige.

# Ok let's read our data with `CSV.jl` and output into a `DataFrame` from `DataFrames.jl`:
using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/duncan.csv"
duncan = CSV.read(HTTP.get(url).body, DataFrame)
describe(duncan)

# As you can see from the `describe()` output the average occupation's percentage of respondents who classified their occupation
# as at least "good" with respect to prestige is around 41%. But `prestige` variable is very dispersed and actually has a bimodal
# distribution:

@df duncan density(:prestige, label=false)
savefig(joinpath(@OUTPUT, "prestige_density.svg")); # hide

# \fig{prestige_density}
# \center{*Density Plot of `prestige`*} \\

# Besides that, the mean `prestige` per `type` shows us where the source of variation might come from:

gdf_type = groupby(duncan, :type)
@df combine(gdf_type, :prestige => mean) bar(:type, :prestige_mean, label=false)
savefig(joinpath(@OUTPUT, "prestige_per_type.svg")); # hide

# \fig{prestige_per_type}
# \center{*Mean `prestige` per `type`*} \\

# Now let's us instantiate our model with the data:

X = Matrix(select(duncan, [:income, :education]))
y = duncan[:, :prestige]
model = robustreg(X, y);

# And, finally, we will sample from the Turing model. We will be using the default `NUTS()` sampler with `2_000` samples, with
# 4 Markov chains using multiple threads `MCMCThreads()`:

chain = sample(model, NUTS(), MCMCThreads(), 2_000, 4)
summarystats(chain)

# We had no problem with the Markov chains as all the `rhat` are well below `1.01` (or above `0.99`).
# Also note that all degrees of freedom parameters, the `ν` stuff, have been estimated with mean around 3 to 5,
# which indeed signals that our model needed fat tails to make a robust inference.
# Our model has an error `σ` of around 7. So it estimates occupation's prestige ±7. The intercept `α` is the basal occupation's
# prestige value. So each occupation has -7±7 prestige before we add the coefficients multiplied by the occupations' independent variables.
# And from our coefficients $\boldsymbol{\beta}$, we can see that the `quantile()` tells us the uncertainty around their
# estimates:

quantile(chain)

# * `β[1]` -- first column of `X`, `income`, has 95% credible interval from 0.55 to 0.96. This means that an increase of U\$ 1,000 in occupations' annual income is associated with an increase in roughly 0.5 to 1.0 in occupation's prestige.
# * `β[2]` -- second column of `X`, `education`, has a 95% credible interval from 0.28 to 0.61. So we expect that an increase of 1% in occupations' percentage of respondents who had a high school diploma increases occupations' prestige roughly 0.3 to 0.6.

# That's how you interpret 95% credible intervals from a `quantile()` output of a robust regression `Chains` object.

# ## References

# Duncan, O. D. (1961). A socioeconomic index for all occupations. Class: Critical Concepts, 1, 388–426.
