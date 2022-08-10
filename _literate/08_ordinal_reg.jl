# # Bayesian Ordinal Regression

# Leaving the universe of linear models, we start to venture into generalized linear models (GLM). The second is
# **ordinal regression**.

# A ordinal regression behaves exactly like a linear model: it makes a prediction simply by computing a weighted
# sum of the independent variables $\mathbf{X}$ by the estimated coefficients $\boldsymbol{\beta}$,
# but now we have not only one intercept but several intercepts $\alpha_k$ for $k \in K$.

# We use ordinal regression when our dependent variable is ordinal.
# That means it has different that have a "natural order"**.
# Most important, the distance between values is not the same.
# For example, imagine a pain score scale that goes from 1 to 10.
# The distance between 1 and 2 is different from the distance 9 to 10.
# Another example is opinion pools with their ubiquously disagree-agree range
# of plausible values.
# These are also known as Likert scale variables.
# The distance between "disagree" to "not agree or disagree" is different
# than the distance between "agree" and "strongly agree".

# This assumption is what we call the "metric" assumption,
# also called as the equidistant assumption.
# Almost always when we model an ordinal dependent variable this assumption is violated.
# Thus, we cannot blindly employ linear regression here.

# ## How to deal with Ordered Discrete Dependent Variable?

# So, how we deal with ordered discrete responses in our dependent variable?
# This is similar with the previous logistic regression approach.
# We have to do a **non-linear transformation of the dependent variable**.

# ### Cumulative Distribution Function (CDF)

# In the case of **ordinal regression**, we need to first transform the dependent variable into a **cumulative scale**.
# We need to calculate the **cumulative distribution function** (CDF) of our dependent variable:

# $$P(Y \leq y) = \sum^y_{i=y_{\text{min}}} P(Y = i)$$

# The **CDF is a monotonic increasing function** that depicts the **probability of our random variable $Y$ taking values less than a certain value**.
# In our case, the discrete ordinal case, these values can be represented as positive integers ranging from 1 to the length of possible values.
# For instance, a 6-categorical ordered response variable will have 6 possible values, and all their CDFs will lie between 0 and 1.
# Furthermore, their sum will be 1; since it represents the total probability of the variable taking any of the possible values, i.e. 100%.

# ### Log-cumulative-odds

# That is still not enough, we need to apply the **logit function to the CDF**:

# $$\mathrm{logit}(x) = \mathrm{logistic}^{-1}(x) = \ln\left(\frac{x}{1 -x}\right)$$

# where $\ln$ is the natural logarithm function.

# The logit is the **inverse of the logistic transformation**,
# it takes as a input any number between 0 and 1
# (where a probability is the perfect candidate) and outputs a real number,
# which we call the **log-odds**.

# Since we are taking the log-odds of the CDF, we can call this complete transformation as
# **log-odds of the CDF**, or **log-cumulative-odds**.

# ### $K-1$ Intercepts

# Now, the next question is: what do we do with the log-cumulative-odds?

# **We need the log-cumulative-odds because it allows us to construct different intercepts for the possible values our ordinal dependent variable**.
# We create an unique intercept for each possible outcome $k \in K$.

# Notice that the highest probable value of $Y$ will always have a log-cumulative-odds of $\infty$, since for $p=1$:

# $$\ln \frac{p}{1-p} = \ln \frac{1}{1-1} = \ln 0 = \infty$$

# Thus, we only need $K-1$ intercepts for a $K$ possible depedent variables' response values.
# These are known as **cut points**.

# Each intercept implies a CDF for each value $K$.
# This allows us to **violate the equidistant assumption** absent in most ordinal variables.

# Each intercept implies a log-cumulative-odds for each $k \in K$.
# We also need to **undo the cumulative nature of the $K-1$ intercepts**.
# We can accomplish this by first converting a **log-cumulative-odds back to a cumulative probability**.
# This is done by undoing the logit transformation and applying the logistic function:

# $$\mathrm{logit}^{-1}(x) = \mathrm{logistic}(x) = \frac{1}{1 + e^{-x}}$$

# Then, finally, we can remove the **cumulative from "cumulative probability"** by
# **subtraction of each of the $k$ cut points by their previous $k-1$ cut point**:

# $$P(Y=k) = P(Y \leq k) - P(Y \leq k-1)$$

# where $Y$ is the depedent variable and $k \in K$ are the cut points for each intercept.

# Let me show you an example with some syntethic data.

using DataFrames
using CairoMakie
using AlgebraOfGraphics
using Distributions
using StatsFuns: logit

# Here we have a discrete variable `x` with 6 possible ordered values as response.
# The values range from 1 to 6 having probability, respectively:
# 10%, 15%, 33%, 25%, 10%, and 7%;
# represented with the `probs` vector.
# The underlying distribution is represented by a
# `Categorical` distribution from `Distributions.jl`,
# which takes a vector of probabilities as parameters (`probs`).

# For each value we are calculating:

# 1. **P**robability **M**ass **F**unction with the `pdf` function
# 2. **C**umulative **D**istribution **F**unction with the `cdf` function
# 3. **Log-cumulative-odds** with the `logit` transformation of the CDF

# In the plot below there are 3 subplots:
# - Upper corner: histogram of `x`
# - Left lower corner: CDF of `x`
# - Right lower corner: log-cumulative-odds of `x`

let
    probs = [0.10, 0.15, 0.33, 0.25, 0.10, 0.07]
    dist = Categorical(probs)
    x = 1:length(probs)
    x_pmf=pdf.(dist, x)
    x_cdf=cdf.(dist, x)
    x_logodds_cdf=logit.(x_cdf)
    df = DataFrame(;
        x,
        x_pmf,
        x_cdf,
        x_logodds_cdf)
    labels = ["CDF", "Log-cumulative-odds"]
    fig = Figure()
    plt1 = data(df) *
        mapping(:x, :x_pmf) *
        visual(BarPlot)
    plt2 = data(df) *
        mapping(:x,
                [:x_cdf, :x_logodds_cdf];
                col=dims(1) => renamer(labels)) *
        visual(ScatterLines)
    axis=(; xticks=1:6)
    draw!(fig[1, 2:3], plt1; axis)
    draw!(fig[2, 1:4], plt2;
          axis,
          facet=(; linkyaxes=:none))
    fig
    save(joinpath(@OUTPUT, "logodds.svg"), fig); # hide
end

# \fig{logodds}
# \center{*Ordinal Dependent Variable*} \\

# As we can see, we have $K-1$ (in our case $6-1=5$) intercept values in log-cumulative-odds.
# You can carly see that these values they violate the **equidistant assumption**
# for metric response values.
# The spacing between the cut points are not the same, they vary.

# ## Adding Coefficients $\boldsymbol{\beta}$

# Ok, the $K-1$ intercepts $\boldsymbol{\alpha}$ are done.
# Now let's add coefficients to act as covariate effects in our ordinal regression model.

# We transformed everything into log-odds scale so that we can add effects
# (coefficients multiplying a covariate) or basal rates (intercepts) together.
# For each $k \in K-1$, we calculate:

# $$\phi_k = \alpha_k + \beta_i x_i$$

# where $\alpha_k$ is the log-cumulative-odds for the $k \in K-1$ intercepts,
# $\beta_i$ is the coefficient for the $i$th covariate $x$.
# Finally, $\phi_k$ represents the linear predictor for the $k$th intercept.

# Observe that the coefficient $\beta$ is being added to a log-cumulative-odds,
# such that, it will be expressed in a log-cumulative-odds also.

# We can express it in matrix form:

# $$\boldsymbol{\phi} = \boldsymbol{\alpha} + \mathbf{X} \cdot \boldsymbol{\beta}$$

# where $\boldsymbol{\phi}$, $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$ are vectors
# and $\mathbf{X}$ is the data matrix where each row is an observation and each column a covariate.

# This still obeys the ordered constraint on the dependent variable possible values.

# #### How to Interpret Coefficient $\boldsymbol{\beta}$?

# Now, suppose we have found our ideal value for our $\boldsymbol{\beta}$.
# **How we would interpret our $\boldsymbol{\beta}$ estimated values?**

# First, to recap, $\boldsymbol{\beta}$ measures the strength of association between
# the covariate $\mathbf{x}$ and depedent variable $\mathbf{y}$.
# But, $\boldsymbol{\beta}$ is nested inside a non-linear transformation called
# logistic function:

# $$\mathrm{logistic}(\boldsymbol{\beta}) = \frac{1}{1 + e^{-\boldsymbol{\beta}}}$$

# So, our first step is to **undo the logistic function**.
# There is a function that is called **logit function** that is the **inverse**
# of the logistic function:

# $$\mathrm{logistic}^{-1}(x) = \mathrm{logit}(x) = \ln\left(\frac{x}{1 -x}\right)$$

# where $\ln$ is the natural logarithm function.

# If we analyze closely the logit function we can find that
# inside the $\ln$ there is a disguised odds in the $\frac{x}{1 -x}$.
# Hence, our $\boldsymbol{\beta}$ can be interpreted as the
# **logarithm of the odds**, or in short form: the **log-odds**.

# We already saw how odds, log-odds, and probability are related
# in the previous logistic regression tutorial.
# So you might want to go back there to get the full explanation.

# The log-odds are the key to interpret coefficient $\boldsymbol{\beta}$**.
# Any positive value of $\beta$ means that there exists a positive association between $x$ and $y$, while any negative values indicate a negative association between $x$ and $y$.
# Values close to 0 demonstrates the lack of association between $x$ and $y$.

# ## Likelihood

# We have almost everything we need for our ordinal regression.
# We are only missing a final touch.
# Currently our **logistic function outputs a vector of probabilities** that sums to 1.

# All of the intercepts $\alpha_k$ along with the coefficients $\beta_i$ are in
# log-cumulative-odds scale.
# If we apply the logistic function to the linear predictors $\phi_k$ we get $K-1$
# probabilities: one for each $\phi_k$

# We need a **likelihood that can handle a vector of probabilities and outputs a single
# discrete value**.
# The **categorical distribution is the perfect candidate**.

# ## Bayesian Ordinal Regression

# Now we have all the components for our Bayesian ordinal regression specification:

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Categorical}(\mathbf{p}) \\
# \mathbf{p} &= \text{Logistic}(\boldsymbol{\phi}) \\
# \boldsymbol{\phi} &= \boldsymbol{\alpha} + \mathbf{X} \cdot \boldsymbol{\beta} \\
# \alpha_1 &= \text{CDF}(y_1) \\
# \alpha_k &= \text{CDF}(y_k) - \text{CDF}(y_{k-1}) \quad \text{for} \quad 1 < k < K-1 \\
# \alpha_{K-1} &= 1 - \text{CDF}(y_{K-1})
# \end{aligned}
# $$

# where:

# * $\mathbf{y}$ -- ordered discrete dependent variable.
# * $\mathbf{p}$ -- probability vector of length $K$.
# * $K$ -- number of possible values $\mathbf{y}$ can take, i.e. number of ordered discrete values.
# * $\boldsymbol{\phi}$ -- log-cumulative-odds, i.e. cut points considering the intercepts and covariates effect.
# * $\alpha_k$ -- intercept in log-cumulative-odds for each $k \in K-1$.
# * $\mathbf{X}$ -- covariate data matrix.
# * $\boldsymbol{\beta}$ -- coefficient vector of the same length as the number of columns in $\mathbf{X}$.
# * $\mathrm{logistic}$ -- logistic function.
# * $\mathrm{CDF}$ -- **c**umulative **d**istribution **f**unction.

# What remains is to specify the model parameters' prior distributions:

# * Prior Distribution of $\boldsymbol{\alpha}$ -- Knowledge we possess regarding the model's intercepts.
# * Prior Distribution of $\boldsymbol{\beta}$  -- Knowledge we possess regarding the model's independent variables' coefficients.

# Our goal is to instantiate an ordinal regression with the observed data ($\mathbf{y}$ and $\mathbf{X}$) and find the posterior
# distribution of our model's parameters of interest ($\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$). This means to find the full posterior
# distribution of:

# $$ P(\boldsymbol{\theta} \mid \mathbf{y}) = P(\boldsymbol{\alpha}, \boldsymbol{\beta} \mid \mathbf{y}) $$

# Note that contrary to the linear regression, which used a Gaussian/normal likelihood function, we don't have an error
# parameter $\sigma$ in our ordinal regression. This is due to the Categorical distribution not having
# a "scale" parameter such as the $\sigma$ parameter in the Gaussian/normal distribution.

# This is easily accomplished with Turing:

using Turing
using Bijectors
using LazyArrays
using LinearAlgebra
using Random:seed!
seed!(123)
setprogress!(false) # hide

@model function ordreg(X,  y; predictors=size(X, 2), ncateg=maximum(y))
    #priors
    cutpoints ~ Bijectors.ordered(filldist(TDist(3) * 5, ncateg - 1))
    β ~ filldist(TDist(3) * 2.5, predictors)

    #likelihood
    y ~ arraydist([OrderedLogistic(X[i, :] ⋅ β, cutpoints) for i in 1:length(y)])
end;

# First, let's deal with the new stuff in our model: the **`Bijectors.ordered`**.
# As I've said in the [4. **How to use Turing**](/pages/04_Turing/),
# Turing has a rich ecossystem of packages.
# Bijectors implements a set of functions for transforming constrained random variables
# (e.g. simplexes, intervals) to Euclidean space.
# Here we are defining `cutpoints` as a `ncateg - 1` vector of Student-$t$ distributions
# with mean 0, standard deviation 5 and degrees of freedom $\nu = 3$.
# Remember that we only need $K-1$ cutpoints for all of our $K$ intercepts.
# And we are also contraining it to be an ordered vector with `Bijectors.ordered`,
# such that for all cutpoints $c_i$ we have $c_1 < c_2 < ... c_{k-1}$.

# As before, we are giving $\boldsymbol{\beta}$ a very weakly informative priors of a
# Student-$t$ distribution centered on 0 with variance 1 and degrees of freedom $\nu = 3$.
# That wide-tailed $t$ distribution will cover all possible values for our coefficients.
# Remember the Student-$t$ also has support over all the real number line $\in (-\infty, +\infty)$. Also the `filldist()` is a nice Turing's function which takes any univariate or multivariate distribution and returns another distribution that repeats the input distribution.

# Finally, in the likelihood,
# Turing's `arraydist()` function wraps an array of distributions returning a new distribution sampling from the individual
# distributions.
# And we use some indexing inside an array literal.

# ## Example - Esoph

# For our example, I will use a famous dataset called `esoph` (Breslow & Day, 1980),
# which is data from a case-control study of (o)esophageal cancer in Ille-et-Vilaine, France.
# It has records for 88 age/alcohol/tobacco combinations:

# * `agegp`: Age group
#    * `1`:  25-34 years
#    * `2`:  35-44 years
#    * `3`:  45-54 years
#    * `4`:  55-64 years
#    * `5`:  65-74 years
#    * `6`:  75+ years
# * `alcgp`: Alcohol consumption
#    * `1`:  0-39 g/day
#    * `2`: 40-79 g/day
#    * `3`: 80-119 g/day
#    * `4`: 120+ g/day
# * `tobgp`: Tobacco consumption
#    * `1`: 0-9 g/day
#    * `2`: 10-19 g/day
#    * `3`: 20-29 g/day
#    * `4`: 30+ g/day
# * `ncases`: Number of cases
# * `ncontrols`: Number of controls

# Ok let's read our data with `CSV.jl` and output into a `DataFrame` from `DataFrames.jl`:
using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/esoph.csv"
esoph = CSV.read(HTTP.get(url).body, DataFrame)
describe(esoph)

# As you can see from the `describe()` output 58% of the respondents switched wells and 42% percent of respondents
# somehow are engaged in community organizations. The average years of education of the household's head is approximate
# 5 years and ranges from 0 (no education at all) to 17 years. The distance to safe drinking water is measured in meters
# and averages 48m ranging from less than 1m to 339m. Regarding arsenic levels I cannot comment because the only thing I
# know that it is toxic and you probably would never want to have your well contaminated with it. Here, we believe that all
# of those variables somehow influence the probability of a respondent switch to a safe well.

# Now let's us instantiate our model with the data.
# But here I need to do some data wrangling to create the data matrix `X`.
# Specifically, I need to convert all of the categorical variables to integer values,
# representing the ordinal values of both our independent and also dependent variables:

using CategoricalArrays

transform!(
    esoph,
    :agegp =>
        x -> categorical(
            x; levels=["25-34", "35-44", "45-54", "55-64", "65-74", "75+"], ordered=true
        ),
    :alcgp =>
        x -> categorical(x; levels=["0-39g/day", "40-79", "80-119", "120+"], ordered=true),
    :tobgp =>
        x -> categorical(x; levels=["0-9g/day", "10-19", "20-29", "30+"], ordered=true);
    renamecols=false,
)
transform!(esoph, [:agegp, :alcgp, :tobgp] .=> ByRow(levelcode); renamecols=false)

X = Matrix(select(esoph, [:agegp, :alcgp]))
y = esoph[:, :tobgp]
model = ordreg(X, y);

# And, finally, we will sample from the Turing model. We will be using the default `NUTS()` sampler with `2_000` samples, with
# 4 Markov chains using multiple threads `MCMCThreads()`:

chain = sample(model, NUTS(), MCMCThreads(), 2_000, 4)
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
    select(_,
        :parameters,
        names(_, r"%") .=> ByRow(exp),
        renamecols=false)
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
    select(_,
        :parameters,
        names(_, r"%") .=> ByRow(logodds2prob),
        renamecols=false)
end

# There you go, much better now. Let's analyze our results.
# The `cutpoints` is the basal rate of the probability of our dependent variable
# having values less than a certain value.
# For example the cutpoint for having values less than `2` which its code represents
# the tobacco comsumption of 10-19 g/day has a median value of 20%.

# Now let's take a look at our coefficients
# All coefficients whose 95% credible intervals captures the value $\frac{1}{2} = 0.5$ tells
# that the effect on the propensity of tobacco comsumption is inconclusive.
# It is pretty much similar to a 95% credible interval that captures the 0 in
# the linear regression coefficients.

# That's how you interpret 95% credible intervals from a `quantile()` output of a ordinal regression `Chains`
# object converted from log-odds to probability.

# ## Footnotes

# [^logit]: actually the [logit](https://en.wikipedia.org/wiki/Logit) function or the log-odds is the logarithm of the odds $\frac{p}{1-p}$ where $p$ is a probability.

# ## References

# Breslow, N. E. & Day, N. E. (1980). **Statistical Methods in Cancer Research. Volume 1: The Analysis of Case-Control Studies**. IARC Lyon / Oxford University Press.
