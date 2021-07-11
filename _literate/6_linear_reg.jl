# # Bayesian Linear Regression

# > "All models are wrong but some are useful"
# > \\ \\
# > George Box (Box, 1976)

# This tutorial begins with a very provocative quote from the statistician
# [George Box](https://en.wikipedia.org/wiki/George_E._P._Box) (figure below)
# on statistical models. Yes, all models are somehow wrong. But they are very useful.
# The idea is that the reality is too complex for us to understand when analyzing it in
# a naked and raw way. We need to somehow simplify it into individual components and analyze
# their relationships. But there is a danger here: any simplification of reality promotes loss
# of information in some way. Therefore, we always have a delicate balance between simplifications
# of reality through models and the inherent loss of information. Now you ask me:
# "how are they useful?" Imagine that you are in total darkness and you have a very powerful
# flashlight but with a narrow beam. Are you going to throw the flashlight away because it
# can't light everything around you and stay in the dark? You must use the flashlight to aim
# at interesting places in the darkness in order to illuminate them. You will never find a
# flashlight that illuminates everything with the clarity you need to analyze all the fine details
# of reality. Just as you will never find a unique model that will explain the whole reality around
# you. You need different flashlights just like you need different models. Without them you will
# be in total darkness.

# ![George Box](/pages/images/george_box.jpg)
#
# \center{*George Box*} \\

# ## Linear Regression

# Let's talk about a class of model known as linear regression. The idea here is to model a continuous dependent variable
# with a linear combination of independent variables.

# $$ \mathbf{y} = \alpha +  \mathbf{X} \boldsymbol{\beta} + \epsilon \label{linear reg} $$

# where:

# * $\mathbf{y}$ -- dependent variable
# * $\alpha$ -- intercept
# * $\boldsymbol{\beta}$ -- coefficient vector
# * $\mathbf{X}$ -- data matrix
# * $\epsilon$ -- model error

# To estimate the $\boldsymbol{\beta}$ coefficients we use a Gaussian/normal likelihood function.
# Mathematically the Bayesian regression model is:

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Normal}\left( \alpha + \mathbf{X} \cdot \boldsymbol{\beta}, \sigma \right) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \boldsymbol{\beta} &\sim \text{Normal}(\mu_{\boldsymbol{\beta}}, \sigma_{\boldsymbol{\beta}}) \\
# \sigma &\sim \text{Exponential}(\lambda_\sigma)
# \end{aligned}
# $$

# Here we see that the likelihood function $P(\mathbf{y} \mid \boldsymbol{\theta})$ is a normal distribution in which $\mathbf{y}$
# depends on the parameters of the model $\alpha$ and $\boldsymbol{\beta}$, in addition to having an
# error $\sigma$. We condition $\mathbf{y}$ onto the observed data $\mathbf{X}$ by inserting
# $\alpha + \mathbf{X} \cdot \boldsymbol{\beta}$ as the linear predictor of the model (the mean $\mu$ parameter of the
# model's Normal likelihood function, and $\sigma$ is the variance parameter). What remains is to specify which are the
# priors of the model parameters:

# * Prior Distribution of $\alpha$ -- Knowledge we possess regarding the model's intercept.
# * Prior Distribution of $\boldsymbol{\beta}$  -- Knowledge we possess regarding the model's independent variables' coefficients.
# * Prior Distribution of $\sigma$ -- Knowledge we possess regarding the model's error. Important that the error can only be positive. In addition, it is intuitive to place a distribution that gives greater weight to values close to zero, but that also allows values that are far from zero, so a distribution with a long tail is welcome. Candidate distributions are $\text{Exponential}$ which is only supported on positive real numbers (so it solves the question of negative errors) or $\text{Cauchy}^+$ truncated to only positive numbers (remembering that the distribution Cauchy is Student's $t$ with degrees of freedom $\nu = 1$).

# Our goal is to instantiate a linear regression with the observed data ($\mathbf{y}$ and $\mathbf{X}$) and find the posterior
# distribution of our model's parameters of interest ($\alpha$ and $\boldsymbol{\beta}$). This means to find the full posterior
# distribution of:

# $$ P(\boldsymbol{\theta} \mid \mathbf{y}) = P(\alpha, \boldsymbol{\beta}, \sigma \mid \mathbf{y}) $$

# This is easily accomplished with Turing:

using Turing
using Statistics: mean, std
using Random:seed!
seed!(123)
setprogress!(false) # hide

@model linreg(X, y; predictors=size(X, 2)) = begin
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))
    β ~ filldist(TDist(3), predictors)
    σ ~ Exponential(1)

    #likelihood
    y ~ MvNormal(α .+ X * β, σ)
end;

# Here I am specifying very weakly informative priors:

# * $\alpha \sim \text{Normal}(\bar{\mathbf{y}}, 2.5 \cdot \sigma_{\mathbf{y}})$ -- This means a normal distribution centered on `y`'s mean with a standard deviation 2.5 times the standard deviation of `y`. That prior should with ease cover all possible values of $\alpha$. Remember that the normal distribution has support over all the real number line $\in (-\infty, +\infty)$.
# * $\boldsymbol{\beta} \sim \text{Student-}t(0,1,3)$ -- The predictors all have a prior distribution of a Student-$t$ distribution centered on 0 with variance 1 and degrees of freedom $\nu = 3$. That wide-tailed $t$ distribution will cover all possible values for our coefficients. Remember the Student-$t$ also has support over all the real number line $\in (-\infty, +\infty)$. Also the `filldist()` is a nice Turing's function which takes any univariate or multivariate distribution and returns another distribution that repeats the input distribution.
# * $\sigma \sim \text{Exponential}(1)$ -- A wide-tailed-positive-only distribution perfectly suited for our model's error.

# ## Example - Children's IQ Score

# For our example, I will use a famous dataset called `kidiq` (Gelman & Hill, 2007), which is data from a survey of adult American women and their respective children. Dated from 2007, it has 434 observations and 4 variables:

# * `kid_score`: child's IQ
# * `mom_hs`: binary/dummy (0 or 1) if the child's mother has a high school diploma
# * `mom_iq`: mother's IQ
# * `mom_age`: mother's age

# Ok let's read our data with `CSV.jl` and output into a `DataFrame` from `DataFrames.jl`:

using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/kidiq.csv"
kidiq = CSV.read(HTTP.get(url).body, DataFrame)
describe(kidiq)

# As you can see from the `describe()` output, the mean children's IQ is around 87 while the mother's is 100. Also the mother's
# range from 17 to 29 years with mean of around 23 years old. Finally, note that 79% of mothers have a high school diploma.

# Now let's us instantiate our model with the data:

X = Matrix(select(kidiq, Not(:kid_score)))
y = kidiq[:, :kid_score]
model = linreg(X, y);

# And, finally, we will sample from the Turing model. We will be using the default `NUTS()` sampler with `2_000` samples, but
# now we will sample from 4 Markov chains using multiple threads `MCMCThreads()`:

chain = sample(model, NUTS(), MCMCThreads(), 2_000, 4)
summarystats(chain)

# We had no problem with the Markov chains as all the `rhat` are well below `1.01` (or above `0.99`).
# Our model has an error `σ` of around 18. So it estimates IQ±9. The intercept `α` is the basal child's IQ.
# So each child has 22±9 IQ before we add the coefficients multiplied by the child's independent variables.
# And from our coefficients $\boldsymbol{\beta}}$, we can see that the `quantile()` tells us the uncertainty around their
# estimates:

quantile(chain)

# * `β[1]` -- first column of `X`, `mom_hs`, has 95% credible interval that is all over the place, including zero. So its effect on child's IQ is inconclusive.
# * `β[2]` -- second column of `X`, `mom_iq`, has a 95% credible interval from 0.46 to 0.69. So we expect that every increase in the mother's IQ i associated with a 0.46 to 0.69 increase in the child's IQ.
# * `β[3]` -- third column of `X`, `mom_age`, has also 95% credible interval that is all over the place, including zero. Like `mom_hs`, its effect on child's IQ is inconclusive.

# That's how you interpret 95% credible intervals from a `quantile()` output of a linear regression `Chains` object.

# ## References

# Box, G. E. P. (1976). Science and Statistics. Journal of the American Statistical Association, 71(356), 791–799. https://doi.org/10.2307/2286841
#
# Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.
