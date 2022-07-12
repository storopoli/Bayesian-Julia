# # Bayesian Regression with Count Data

# Leaving the universe of linear models, we start to venture into generalized linear models (GLM). The second of these is
# **regression with count data** (also called Poisson regression).

# A regression with count data behaves exactly like a linear model: it makes a prediction simply by computing a weighted
# sum of the independent variables $\mathbf{X}$ by the estimated coefficients $\boldsymbol{\beta}$, plus an intercept
# $\alpha$. However, instead of returning a continuous value $y$, such as linear regression, it returns the **natural
# log** of $y$.

# We use regression with count data when our dependent variable is restricted to positive integers, *i.e.* $y \in \mathbb{Z}^+$.
# See the figure below for a graphical intuition of the exponential function:

using Plots, LaTeXStrings

plot(exp, -6, 6, label=false,
     xlabel=L"x", ylabel=L"e^x")
savefig(joinpath(@OUTPUT, "exponential.svg")); # hide

# \fig{exponential}
# \center{*Exponential Function*} \\

# As we can see, the exponential function is basically a mapping of any real number to a
# positive real number in the range between 0 and $+\infty$ (non-inclusive):

# $$ \text{Exponential}(x) = \{ \mathbb{R} \in [- \infty , + \infty] \} \to \{ \mathbb{R} \in [0, + \infty] \} $$

# That is, the exponential function is the ideal candidate for when we need to convert something continuous without restrictions
# to something continuous restricted to taking positive values only. That is why it is used when we need a model to have a
# positive-only dependent variable. This is the case of a dependent variable for count data.

# ## Comparison with Linear Regression

# Linear regression follows the following mathematical formulation:

# $$ \text{Linear} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n $$

# * $\theta$ - model parameters
#   * $\theta_0$ - intercept
#   * $\theta_1, \theta_2, \dots$ - independent variables $x_1, x_2, \dots$ coefficients
# * $n$ - total number of independent variables

# Regression with count data would add the exponential function to the linear term:

# $$ \log(y) = \theta_0 \cdot \theta_1 x_1 \cdot \theta_2 x_2 \cdot \dots \cdot \theta_n x_n $$

# which is the same as:

# $$ y = e^{(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n)} $$

# ## Bayesian Regression with Count Data

# We can model regression with count data in two ways. The first option with a **Poisson likelihood** function
# and the second option with a **negative binomial likelihood** function.

# With the **Poisson likelihood** we model a discrete and positive dependent variable $y$ by assuming that a given number of
# independent $y$ events will occur with a known constant average rate.

# In a **negative binomial likelihood**, model a discrete and positive dependent variable $y$ by assuming that a given number $n$ of
# independent $y$ events will occur by asking a yes-no question for each $n$ with probability $p$ until $k$ success(es) is obtained.
# Note that it becomes identical to the Poisson likelihood when at the limit of $k \to \infty$. This makes the negative binomial a
# **robust option to replace a Poisson likelihood** to model phenomena with a *overdispersion* (excess expected variation in data).
# This occurs due to the Poisson likelihood making an assumption that the dependent variable $y$ has the same mean and variance,
# while in the negative binomial likelihood the mean and the variance do not need to be equal.

# ### Using Poisson Likelihood

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Poisson}\left( e^{(\alpha + \mathbf{X} \cdot \boldsymbol{\beta})} \right) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \boldsymbol{\beta} &\sim \text{Normal}(\mu_{\boldsymbol{\beta}}, \sigma_{\boldsymbol{\beta}})
# \end{aligned}
# $$

# where:

# * $\mathbf{y}$ -- discrete and positive dependent variable.
# * $e$ -- exponential.
# * $\alpha$ -- intercept.
# * $\boldsymbol{\beta}$ -- coefficient vector.
# * $\mathbf{X}$ -- data matrix.

# As we can see, the linear predictor $\alpha + \mathbf{X} \cdot \boldsymbol{\beta}$ is the logarithm of the value of
# $y$. So we need to apply the exponential function the values of the linear predictor:

# $$
# \begin{aligned}
# \log(\mathbf{y}) &= \alpha + \mathbf{X} \cdot \boldsymbol{\beta} \\
# \mathbf{y} &= e^{\alpha \mathbf{X} \cdot \boldsymbol{\beta}} \\
# \mathbf{y} &= e^{\alpha} \cdot e^{\left( \mathbf{X} \cdot \boldsymbol{\beta} \right) }
# \end{aligned}
# $$

# The intercept $\alpha$ and coefficients $\boldsymbol{\beta}$ are only interpretable after exponentiation.

# What remains is to specify the model parameters' prior distributions:

# * Prior Distribution of $\alpha$ -- Knowledge we possess regarding the model's intercept.
# * Prior Distribution of $\boldsymbol{\beta}$  -- Knowledge we possess regarding the model's independent variables' coefficients.

# Our goal is to instantiate a regression with count data using the observed data ($\mathbf{y}$ and $\mathbf{X}$) and find the posterior
# distribution of our model's parameters of interest ($\alpha$ and $\boldsymbol{\beta}$). This means to find the full posterior
# distribution of:

# $$ P(\boldsymbol{\theta} \mid \mathbf{y}) = P(\alpha, \boldsymbol{\beta} \mid \mathbf{y}) $$

# Note that contrary to the linear regression, which used a Gaussian/normal likelihood function, we don't have an error
# parameter $\sigma$ in our regression with count data. This is due to the Poisson not having
# a "scale" parameter such as the $\sigma$ parameter in the Gaussian/normal distribution.

# This is easily accomplished with Turing:

using Turing
using LazyArrays
using Random:seed!
seed!(123)
setprogress!(false) # hide

@model function poissonreg(X,  y; predictors=size(X, 2))
    #priors
    α ~ Normal(0, 2.5)
    β ~ filldist(TDist(3), predictors)

    #likelihood
    y ~ arraydist(LazyArray(@~ LogPoisson.(α .+ X * β)))
end;

# Here I am specifying very weakly informative priors:

# * $\alpha \sim \text{Normal}(0, 2.5)$ -- This means a normal distribution centered on 0 with a standard deviation of 2.5. That prior should with ease cover all possible values of $\alpha$. Remember that the normal distribution has support over all the real number line $\in (-\infty, +\infty)$.
# * $\boldsymbol{\beta} \sim \text{Student-}t(0,1,3)$ -- The predictors all have a prior distribution of a Student-$t$ distribution centered on 0 with variance 1 and degrees of freedom $\nu = 3$. That wide-tailed $t$ distribution will cover all possible values for our coefficients. Remember the Student-$t$ also has support over all the real number line $\in (-\infty, +\infty)$. Also the `filldist()` is a nice Turing's function which takes any univariate or multivariate distribution and returns another distribution that repeats the input distribution.

# Turing's `arraydist()` function wraps an array of distributions returning a new distribution sampling from the individual
# distributions. And the LazyArrays' `LazyArray()` constructor wrap a lazy object that wraps a computation producing an array
# to an array. Last, but not least, the macro `@~` creates a broadcast and is a nice short hand for the familiar dot `.`
# broadcasting operator in Julia. This is an efficient way to tell Turing that our `y` vector is distributed lazily as a
# `LogPoisson` broadcasted to `α` added to the product of the data matrix `X` and `β` coefficient vector. `LogPoisson` is
# Turing's efficient distribution that already apply exponentiation to all the linear predictors.

# ### Using Negative Binomial Likelihood

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Negative Binomial}\left( e^{(\alpha + \mathbf{X} \cdot \boldsymbol{\beta})}, \phi \right) \\
# \phi &\sim \text{Gamma}(0.01, 0.01) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \boldsymbol{\beta} &\sim \text{Normal}(\mu_{\boldsymbol{\beta}}, \sigma_{\boldsymbol{\beta}})
# \end{aligned}
# $$

# where:

# * $\mathbf{y}$ -- discrete and positive dependent variable.
# * $e$ -- exponential.
# * $\phi$ -- dispersion.
# * $\phi^-$ -- reciprocal dispersion.
# * $\alpha$ -- intercept.
# * $\boldsymbol{\beta}$ -- coefficient vector.
# * $\mathbf{X}$ -- data matrix.

# Note that when we compare with the Poisson model, we have a new parameter $\phi$ that parameterizes the negative binomial
# likelihood. This parameter is the probability of successes $p$ of the negative binomial distribution and we generally use
# a Gamma distribution as prior so that the inverse of $\phi$ which is $\phi^-$ fulfills the function of a "reciprocal dispersion"
# parameter. Most of the time we use a weakly informative prior of the parameters shape $\alpha = 0.01$ and scale $\theta = 0.01$
# (Gelman et al., 2013; 2020). But you can also use $\phi^- \sim \text{Exponential}(1)$ as prior (McElreath, 2020).

# Here is what a $\text{Gamma}(0.01, 0.01)$ looks like:

using StatsPlots, Distributions
plot(Gamma(0.01, 0.01),
        lw=2, label=false,
        xlabel=L"\phi",
        ylabel="Density",
        xlims=(0, 0.001))
savefig(joinpath(@OUTPUT, "gamma.svg")); # hide

# \fig{gamma}
# \center{*Gamma Distribution with $\alpha = 0.01$ and $\theta = 0.01$*} \\

# In both likelihood options, what remains is to specify the model parameters' prior distributions:

# * Prior Distribution of $\alpha$ -- Knowledge we possess regarding the model's intercept.
# * Prior Distribution of $\boldsymbol{\beta}$  -- Knowledge we possess regarding the model's independent variables' coefficients.

# Our goal is to instantiate a regression with count data using the observed data ($\mathbf{y}$ and $\mathbf{X}$) and find the posterior
# distribution of our model's parameters of interest ($\alpha$ and $\boldsymbol{\beta}$). This means to find the full posterior
# distribution of:

# $$ P(\boldsymbol{\theta} \mid \mathbf{y}) = P(\alpha, \boldsymbol{\beta} \mid \mathbf{y}) $$

# Note that contrary to the linear regression, which used a Gaussian/normal likelihood function, we don't have an error
# parameter $\sigma$ in our regression with count data. This is due to neither the Poisson nor negative binomial distributions having
# a "scale" parameter such as the $\sigma$ parameter in the Gaussian/normal distribution.

# #### Alternative Negative Binomial Parameterization

# One last thing before we get into the details of the negative binomial distribution is to consider an alternative parameterization.
# Julia's `Distributions.jl` and, consequently, Turing's parameterization of the negative binomial distribution follows the following the [Wolfram reference](https://reference.wolfram.com/language/ref/NegativeBinomialDistribution.html):

# $$
# \text{Negative-Binomial}(k \mid r, p) \sim \frac{\Gamma(k+r)}{k! \Gamma(r)} p^r (1 - p)^k, \quad \text{for } k = 0, 1, 2, \ldots
# $$

# where:
# * $k$ -- number of failures before the $r$th success in a sequence of independent Bernoulli trials
# * $r$ -- number of successes
# * $p$ -- probability of success in an individual Bernoulli trial

# This is not ideal for most of the modeling situations that we would employ the negative binomial distribution.
# In particular, we want to have a parameterization that is more appropriate for count data.
# What we need is the familiar **mean** (or location) and **variance** (or scale) parameterization.
# If we look in [Stan's documentation for the `neg_binomial_2` function](https://mc-stan.org/docs/functions-reference/nbalt.html), we have the following two equations:

# $$
# \begin{aligned}
# \mu &= \frac{r (1 - p)}{p} \label{negbin_mean} \\
# \mu + \frac{\mu^2}{\phi} &= \frac{r (1 - p)}{p^2}
# \end{aligned}
# $$

# With a little bit of algebra, we can substitute the first equation of \eqref{negbin_mean} into the right hand side of the second equation and get the following:

# $$
# \begin{aligned}
# \mu + \frac{\mu^2}{\phi} &= \frac{μ}{p} \\
# 1 + \frac{\mu}{\phi} &= \frac{1}{p} \\
# p &= \frac{1}{\frac{1 + \mu}{\phi}}
# \end{aligned}
# $$

# Then in \eqref{negbin_mean} we have:

# $$
# \begin{aligned}
# \mu &= r \left(1 - \left( \frac{1}{\frac{1 + \mu}{\phi}} \right) \right) \cdot \left(1 + \frac{\mu}{\phi} \right) \\
# \mu &= r \left( \left(1 + \frac{\mu}{\phi} \right) - 1 \right) \\
# r &= \phi
# \end{aligned}
# $$

# Hence, the resulting map is $\text{Negative-Binomial}(\mu, \phi) \equiv \text{Negative-Binomial} \left( r = \phi, p = \frac{1}{\frac{1 + \mu}{\phi}} \right)$.
# I would like to point out that this implementation was done by [Tor Fjelde](https://github.com/torfjelde) in a [COVID-19 model with the code available in GitHub](https://github.com/cambridge-mlg/Covid19).
# So we can use this parameterization in our negative binomial regression model.
# But first, we need to define an alternative negative binomial distribution function:

function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    p = p > 0 ? p : 1e-4 # numerical stability
    r = ϕ

    return NegativeBinomial(r, p)
end

# Now we create our Turing model with the alternative `NegBinomial2` parameterization:

@model function negbinreg(X,  y; predictors=size(X, 2))
    #priors
    α ~ Normal(0, 2.5)
    β ~ filldist(TDist(3), predictors)
    ϕ⁻ ~ Gamma(0.01, 0.01)
    ϕ = 1 / ϕ⁻

    #likelihood
    y ~ arraydist(LazyArray(@~ NegativeBinomial2.(exp.(α .+ X * β), ϕ)))
end;

# Here I am also specifying very weakly informative priors:

# * $\alpha \sim \text{Normal}(0, 2.5)$ -- This means a normal distribution centered on 0 with a standard deviation of 2.5. That prior should with ease cover all possible values of $\alpha$. Remember that the normal distribution has support over all the real number line $\in (-\infty, +\infty)$.
# * $\boldsymbol{\beta} \sim \text{Student-}t(0,1,3)$ -- The predictors all have a prior distribution of a Student-$t$ distribution centered on 0 with variance 1 and degrees of freedom $\nu = 3$. That wide-tailed $t$ distribution will cover all possible values for our coefficients. Remember the Student-$t$ also has support over all the real number line $\in (-\infty, +\infty)$. Also the `filldist()` is a nice Turing's function which takes any univariate or multivariate distribution and returns another distribution that repeats the input distribution.
# * $\phi \sim \text{Exponential}(1)$ -- overdispersion parameter of the negative binomial distribution.

# Turing's `arraydist()` function wraps an array of distributions returning a new distribution sampling from the individual
# distributions. And the LazyArrays' `LazyArray()` constructor wrap a lazy object that wraps a computation producing an array
# to an array. Last, but not least, the macro `@~` creates a broadcast and is a nice short hand for the familiar dot `.`
# broadcasting operator in Julia. This is an efficient way to tell Turing that our `y` vector is distributed lazily as a
# `NegativeBinomial2` broadcasted to `α` added to the product of the data matrix `X` and `β` coefficient vector.
# Note that `NegativeBinomial2` does not apply exponentiation so we had to include the `exp.()` broadcasted function to all the linear predictors.

# ## Example - Roaches Extermination

# For our example, I will use a famous dataset called `roaches` (Gelman & Hill, 2007), which is data on the efficacy of a
# pest management system at reducing the number of roaches in urban apartments.
# It has 262 observations and the following variables:

# * `y` -- number of roaches caught.
# * `roach1` -- pretreatment number of roaches.
# * `treatment` -- binary/dummy (0 or 1) for treatment indicator.
# * `senior` -- binary/dummy (0 or 1) for only elderly residents in building.
# * `exposure2` -- number of days for which the roach traps were used

# Ok let's read our data with `CSV.jl` and output into a `DataFrame` from `DataFrames.jl`:
using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/roaches.csv"
roaches = CSV.read(HTTP.get(url).body, DataFrame)
describe(roaches)

# As you can see from the `describe()` output the average number of roaches caught by the pest management system is around 26 roaches.
# The average number of roaches pretreatment is around 42 roaches (oh boy...).
# 30% of the buildings has only elderly residents and 60% of the buildings received a treatment by the pest management system.
# Also note that the traps were set in general for only 1 day and it ranges from 0.2 days (almost 5 hours) to 4.3 days
# (which is approximate 4 days and 7 hours).

# ### Poisson Regression

# Let's first run the Poisson regression.
# First, we instantiate our model with the data:

X = Matrix(select(roaches, Not(:y)))
y = roaches[:, :y]
model_poisson = poissonreg(X, y);

# And, finally, we will sample from the Turing model. We will be using the default `NUTS()` sampler with `2_000` samples, with
# 4 Markov chains using multiple threads `MCMCThreads()`:

chain_poisson = sample(model_poisson, NUTS(), MCMCThreads(), 2_000, 4)
summarystats(chain_poisson)

# We had no problem with the Markov chains as all the `rhat` are well below `1.01` (or above `0.99`).
# Note that the coefficients are in log scale. So we need to apply the exponential function to them.
# We can do this with a transformation in a `DataFrame` constructed from a `Chains` object:

using Chain

@chain quantile(chain_poisson) begin
    DataFrame
    select(_,
        :parameters,
        names(_, r"%") .=> ByRow(exp),
        renamecols=false)
end

# Let's analyze our results. The intercept `α` is the basal number of roaches caught `y` and has
# a median value of 19.4 roaches caught. The remaining 95% credible intervals for the `β`s can be interpreted as follows:

# * `β[1]` -- first column of `X`, `roach1`, has 95% credible interval 1.01 to 1.01. This means that each **increase** in one unit of `roach1` is related to an increase of 1% more roaches caught.
# * `β[2]` -- second column of `X`, `treatment`, has 95% credible interval 0.57 to 0.63. This means that if an apartment was treated with the pest management system then we expect an **decrease** of around 40% roaches caught.
# * `β[3]` -- third column of `X`, `senior`, has a 95% credible interval from 0.64 to 0.73. This means that if an apartment building has only elderly residents then we expect an **decrease** of around 30% roaches caught.
# * `β[4]` -- fourth column of `X`, `exposure2`, has a 95% credible interval from 1.09 to 1.26. Each increase in one day for the exposure of traps in an apartment we expect an **increase** of between 9% to 26% roaches caught.

# That's how you interpret 95% credible intervals from a `quantile()` output of a regression with count data `Chains`
# object converted from a log scale.

# ### Negative Binomial Regression

# Let's now run the negative binomial regression.

model_negbin = negbinreg(X, y);

# We will also default `NUTS()` sampler with `2_000` samples, with 4 Markov chains using multiple threads `MCMCThreads()`:

seed!(111) # hide
chain_negbin = sample(model_negbin, NUTS(),MCMCThreads(), 2_000, 4)
summarystats(chain_negbin)

# We had no problem with the Markov chains as all the `rhat` are well below `1.01` (or above `0.99`).
# Note that the coefficients are in log scale. So we need to also apply the exponential function as we did before.

@chain quantile(chain_negbin) begin
    DataFrame
    select(_,
        :parameters,
        names(_, r"%") .=> ByRow(exp),
        renamecols=false)
end

# Our results show much more uncertainty in the coefficients than in the Poisson regression.
# So it might be best to use the Poisson regression in the `roaches` dataset.

# ## References

# Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
#
# Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis. Chapman and Hall/CRC.
#
# Gelman, A., Hill, J., & Vehtari, A. (2020). Regression and other stories. Cambridge University Press.
#
# McElreath, R. (2020). *Statistical rethinking: A Bayesian course with examples in R and Stan*. CRC press.
