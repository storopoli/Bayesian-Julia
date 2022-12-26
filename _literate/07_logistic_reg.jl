# # Bayesian Logistic Regression

# Leaving the universe of linear models, we start to venture into generalized linear models (GLM). The first is
# **logistic regression** (also called binomial regression).

# A logistic regression behaves exactly like a linear model: it makes a prediction simply by computing a weighted
# sum of the independent variables $\mathbf{X}$ by the estimated coefficients $\boldsymbol{\beta}$, plus an intercept
# $\alpha$. However, instead of returning a continuous value $y$, such as linear regression, it returns the **logistic
# function** of $y$:

# $$ \text{Logistic}(x) = \frac{1}{1 + e^{(-x)}} $$

# We use logistic regression when our dependent variable is binary. It has only two distinct values, usually
# encoded as $0$ or $1$. See the figure below for a graphical intuition of the logistic function:

using Plots, LaTeXStrings

function logistic(x)
    return 1 / (1 + exp(-x))
end

plot(logistic, -10, 10; label=false, xlabel=L"x", ylabel=L"\mathrm{Logistic}(x)")
savefig(joinpath(@OUTPUT, "logistic.svg")); # hide

# \fig{logistic}
# \center{*Logistic Function*} \\

# As we can see, the logistic function is basically a mapping of any real number to a
# real number in the range between 0 and 1:

# $$ \text{Logistic}(x) = \{ \mathbb{R} \in [- \infty , + \infty] \} \to \{ \mathbb{R} \in (0, 1) \} $$

# That is, the logistic function is the ideal candidate for when we need to convert something continuous without restrictions
# to something continuous restricted between 0 and 1. That is why it is used when we need a model to have a probability as a
# dependent variable (remembering that any real number between 0 and 1 is a valid probability). In the case of a binary dependent
# variable, we can use this probability as the chance of the dependent variable taking a value of 0 or 1.

# ## Comparison with Linear Regression

# Linear regression follows the following mathematical formulation:

# $$ \text{Linear} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n $$

# * $\theta$ - model parameters
#   * $\theta_0$ - intercept
#   * $\theta_1, \theta_2, \dots$ - independent variables $x_1, x_2, \dots$ coefficients
# * $n$ - total number of independent variables

# Logistic regression would add the logistic function to the linear term:

# * $\hat{p} = \text{Logistic}(\text{Linear}) = \frac{1}{1 + e^{-\operatorname{Linear}}}$ - predicted probability of the observation being the value 1
# * $\hat{\mathbf{y}}=\left\{\begin{array}{ll} 0 & \text { if } \hat{p} < 0.5 \\ 1 & \text { if } \hat{p} \geq 0.5 \end{array}\right.$ - predicted discrete value of $\mathbf{y}$

# **Example**:

# $$ \text{Probability of Death} = \text{Logistic} \big(-10 + 10 \cdot \text{cancer} + 12 \cdot \text{diabetes} + 8 \cdot \text{obesity} \big) $$

# ## Bayesian Logistic Regression

# We can model logistic regression in two ways. The first option with a **Bernoulli likelihood** function and the second option with
# a **binomial likelihood** function.

# With the **Bernoulli likelihood** we model a binary dependent variable $y$ which is the result of a Bernoulli trial with
# a certain probability $p$.

# In a **binomial likelihood**, we model a continuous dependent variable $y$ which is the number of successes of $n$
# independent Bernoulli trials.

# ### Using Bernoulli Likelihood

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Bernoulli}\left( p \right) \\
# \mathbf{p} &\sim \text{Logistic}(\alpha + \mathbf{X} \cdot \boldsymbol{\beta}) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \boldsymbol{\beta} &\sim \text{Normal}(\mu_{\boldsymbol{\beta}}, \sigma_{\boldsymbol{\beta}})
# \end{aligned}
# $$

# where:

# * $\mathbf{y}$ -- binary dependent variable.
# * $\mathbf{p}$ -- probability of $\mathbf{y}$ taking the value of $\mathbf{y}$ -- success of an independent Bernoulli trial.
# * $\text{Logistic}$ -- logistic function.
# * $\alpha$ -- intercept.
# * $\boldsymbol{\beta}$ -- coefficient vector.
# * $\mathbf{X}$ -- data matrix.

# ### Using Binomial Likelihood

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Binomial}\left( n, p \right) \\
# \mathbf{p} &\sim \text{Logistic}(\alpha + \mathbf{X} \cdot \boldsymbol{\beta}) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \boldsymbol{\beta} &\sim \text{Normal}(\mu_{\boldsymbol{\beta}}, \sigma_{\boldsymbol{\beta}})
# \end{aligned}
# $$

# where:

# * $\mathbf{y}$ -- binary dependent variable -- successes of $n$ independent Bernoulli trials.
# * $n$ -- number of independent Bernoulli trials.
# * $\mathbf{p}$ -- probability of $\mathbf{y}$ taking the value of $\mathbf{y}$ -- success of an independent Bernoulli trial.
# * $\text{Logistic}$ -- logistic function.
# * $\alpha$ -- intercept.
# * $\boldsymbol{\beta}$ -- coefficient vector.
# * $\mathbf{X}$ -- data matrix.

# In both likelihood options, what remains is to specify the model parameters' prior distributions:

# * Prior Distribution of $\alpha$ -- Knowledge we possess regarding the model's intercept.
# * Prior Distribution of $\boldsymbol{\beta}$  -- Knowledge we possess regarding the model's independent variables' coefficients.

# Our goal is to instantiate a logistic regression with the observed data ($\mathbf{y}$ and $\mathbf{X}$) and find the posterior
# distribution of our model's parameters of interest ($\alpha$ and $\boldsymbol{\beta}$). This means to find the full posterior
# distribution of:

# $$ P(\boldsymbol{\theta} \mid \mathbf{y}) = P(\alpha, \boldsymbol{\beta} \mid \mathbf{y}) $$

# Note that contrary to the linear regression, which used a Gaussian/normal likelihood function, we don't have an error
# parameter $\sigma$ in our logistic regression. This is due to neither the Bernoulli nor binomial distributions having
# a "scale" parameter such as the $\sigma$ parameter in the Gaussian/normal distribution.

# Also note that the Bernoulli distribution is a special case of the binomial distribution where $n = 1$:

# $$ \text{Bernoulli}(p) = \text{Binomial}(1, p) $$

# This is easily accomplished with Turing:

using Turing
using LazyArrays
using Random: seed!
seed!(123)
setprogress!(false) # hide

@model function logreg(X, y; predictors=size(X, 2))
    #priors
    α ~ Normal(0, 2.5)
    β ~ filldist(TDist(3), predictors)

    #likelihood
    return y ~ arraydist(LazyArray(@~ BernoulliLogit.(α .+ X * β)))
end;

# Here I am specifying very weakly informative priors:

# * $\alpha \sim \text{Normal}(0, 2.5)$ -- This means a normal distribution centered on 0 with a standard deviation of 2.5. That prior should with ease cover all possible values of $\alpha$. Remember that the normal distribution has support over all the real number line $\in (-\infty, +\infty)$.
# * $\boldsymbol{\beta} \sim \text{Student-}t(0,1,3)$ -- The predictors all have a prior distribution of a Student-$t$ distribution centered on 0 with variance 1 and degrees of freedom $\nu = 3$. That wide-tailed $t$ distribution will cover all possible values for our coefficients. Remember the Student-$t$ also has support over all the real number line $\in (-\infty, +\infty)$. Also the `filldist()` is a nice Turing's function which takes any univariate or multivariate distribution and returns another distribution that repeats the input distribution.

# Turing's `arraydist()` function wraps an array of distributions returning a new distribution sampling from the individual
# distributions. And the LazyArrays' `LazyArray()` constructor wrap a lazy object that wraps a computation producing an array
# to an array. Last, but not least, the macro `@~` creates a broadcast and is a nice short hand for the familiar dot `.`
# broadcasting operator in Julia. This is an efficient way to tell Turing that our `y` vector is distributed lazily as a
# `BernoulliLogit` broadcasted to `α` added to the product of the data matrix `X` and `β` coefficient vector.

# If your dependent variable `y` is continuous and represents the number of successes of $n$ independent Bernoulli trials
# you can use the binomial likelihood in your model:

# ```julia
# y ~ arraydist(LazyArray(@~ BinomialLogit.(n, α .+ X * β)))
# ```

# ## Example - Contaminated Water Wells

# For our example, I will use a famous dataset called `wells` (Gelman & Hill, 2007), which is data from a survey of 3,200
# residents in a small area of Bangladesh suffering from arsenic contamination of groundwater. Respondents with elevated
# arsenic levels in their wells had been encouraged to switch their water source to a safe public or private well in the nearby
# area and the survey was conducted several years later to learn which of the affected residents had switched wells.
# It has 3,200 observations and the following variables:

# * `switch` -- binary/dummy (0 or 1) for well-switching.
# * `arsenic` -- arsenic level in respondent's well.
# * `dist` -- distance (meters) from the respondent's house to the nearest well with safe drinking water.
# * `association` -- binary/dummy (0 or 1) if member(s) of household participate in community organizations.
# * `educ` -- years of education (head of household).

# Ok let's read our data with `CSV.jl` and output into a `DataFrame` from `DataFrames.jl`:
using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/wells.csv"
wells = CSV.read(HTTP.get(url).body, DataFrame)
describe(wells)

# As you can see from the `describe()` output 58% of the respondents switched wells and 42% percent of respondents
# somehow are engaged in community organizations. The average years of education of the household's head is approximate
# 5 years and ranges from 0 (no education at all) to 17 years. The distance to safe drinking water is measured in meters
# and averages 48m ranging from less than 1m to 339m. Regarding arsenic levels I cannot comment because the only thing I
# know that it is toxic and you probably would never want to have your well contaminated with it. Here, we believe that all
# of those variables somehow influence the probability of a respondent switch to a safe well.

# Now let's us instantiate our model with the data:

X = Matrix(select(wells, Not(:switch)))
y = wells[:, :switch]
model = logreg(X, y);

# And, finally, we will sample from the Turing model. We will be using the default `NUTS()` sampler with `1_000` samples, with
# 4 Markov chains using multiple threads `MCMCThreads()`:

chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain)

# We had no problem with the Markov chains as all the `rhat` are well below `1.01` (or above `0.99`).
# Note that the coefficients are in log-odds scale. They are the natural log of the odds[^logit], and odds is
# defined as:

# $$ \text{odds} = \frac{p}{1-p} $$

# where $p$ is a probability. So log-odds is defined as:

# $$ \log(\text{odds}) = \log \left( \frac{p}{1-x} \right) $$

# So in order to get odds from a log-odds we must undo the log operation with a exponentiation.
# This translates to:

# $$ \text{odds} = \exp ( \log ( \text{odds} )) $$

# We can do this with a transformation in a `DataFrame` constructed from a `Chains` object:

using Chain

@chain quantile(chain) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(exp); renamecols=false)
end

# Our interpretation of odds is the same as in betting games. Anything below 1 signals a unlikely probability that $y$ will be $1$.
# And anything above 1 increases the probability of $y$ being $1$, while 1 itself is a neutral odds for $y$ being either $1$ or $0$.
# Since I am not a gambling man, let's talk about probabilities. So I will create a function called `logodds2prob()` that converts
# log-odds to probabilities:

function logodds2prob(logodds::Float64)
    return exp(logodds) / (1 + exp(logodds))
end

@chain quantile(chain) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(logodds2prob); renamecols=false)
end

# There you go, much better now. Let's analyze our results. The intercept `α` is the basal `switch` probability which has
# a median value of 46%. All coefficients whose 95% credible intervals captures the value $\frac{1}{2} = 0.5$ tells
# that the effect on the propensity of `switch` is inconclusive. It is pretty much similar to a 95% credible interval
# that captures the 0 in the linear regression coefficients. So this rules out `β[3]` which is the third column of `X`
# -- `assoc`. The other remaining 95% credible intervals can be interpreted as follows:

# * `β[1]` -- first column of `X`, `arsenic`, has 95% credible interval 0.595 to 0.634. This means that each increase in one unit of `arsenic` is related to an increase of 9.6% to 13.4% propension of `switch` being 1.
# * `β[2]` -- second column of `X`, `dist`, has a 95% credible interval from 0.497 to 0.498. So we expect that each increase in one meter of `dist` is related to a decrease of 0.1% propension of `switch` being 0.
# * `β[4]` -- fourth column of `X`, `educ`, has a 95% credible interval from 0.506 to 0.515. Each increase in one year of `educ` is related to an increase of 0.6% to 1.5% propension of `switch` being 1.

# That's how you interpret 95% credible intervals from a `quantile()` output of a logistic regression `Chains`
# object converted from log-odds to probability.

# ## Footnotes

# [^logit]: actually the [logit](https://en.wikipedia.org/wiki/Logit) function or the log-odds is the logarithm of the odds $\frac{p}{1-p}$ where $p$ is a probability.

# ## References

# Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
