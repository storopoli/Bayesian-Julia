# # Multilevel Models (a.k.a. Hierarchical Models)

# Bayesian hierarchical models (also called multilevel models) are a statistical model written at *multiple* levels
# (hierarchical form) that estimates the parameters of the posterior distribution using the Bayesian approach.
# The sub-models combine to form the hierarchical model, and Bayes' theorem is used to integrate them with the
# observed data and to account for all the uncertainty that is present. The result of this integration is the
# posterior distribution, also known as an updated probability estimate, as additional evidence of the likelihood
# function is integrated together with the prior distribution of the parameters.

# Hierarchical modeling is used when information is available at several different levels of observation units.
# The hierarchical form of analysis and organization helps to understand multiparameter problems and also plays
# an important role in the development of computational strategies.

# Hierarchical models are mathematical statements that involve several parameters, so that the estimates of some parameters
# depend significantly on the values of other parameters. The figure below shows a hierarchical model in which there is a
# $\phi$ hyperparameter that parameterizes the parameters $\theta_1, \theta_2, \dots, \theta_N$ that are finally used to
# infer the posterior density of some variable of interest $\mathbf{y} = y_1, y_2, \dots, y_N$.

# ![Bayesian Workflow](/pages/images/hierarchical.png)
#
# \center{*Hierarchical Model*} \\

# ## When to use Multilevel Models?

# Multilevel models are particularly suitable for research projects where participant data is organized at more than one level, *i.e.* nested data.
# Units of analysis are usually individuals (at a lower level) that are nested in contextual/aggregate units (at a higher level).
# An example is when we are measuring the performance of individuals and we have additional information about belonging to different
# groups such as sex, age group, hierarchical level, educational level or housing status.

# There is a main assumption that cannot be violated in multilevel models which is **exchangeability** (de Finetti, 1974; Nau, 2001).
# Yes, this is the same assumption that we discussed in [2. **What is Bayesian Statistics?**](/pages/2_bayes_stats/).
# This assumption assumes that groups are exchangeable. The figure below shows a graphical representation of the exchangeability.
# The groups shown as "cups" that contain observations shown as "balls". If in the model's inferences, this assumption is violated,
# then multilevel models are not appropriate. This means that, since there is no theoretical justification to support exchangeability,
# the inferences of the multilevel model are not robust and the model can suffer from several pathologies and should not be used for any
# scientific or applied analysis.

# ![Bayesian Workflow](/pages/images/exchangeability-1.png)
# ![Bayesian Workflow](/pages/images/exchangeability-2.png)
#
# \center{*Exchangeability -- Images from [Michael Betancourt](https://betanalpha.github.io/)*} \\

# ## Hyperpriors

# As the priors of the parameters are sampled from another prior of the hyperparameter (upper-level's parameter),
# which are called **hyperpriors**. This makes one group's estimates help the model to better estimate the other groups
# by providing more **robust and stable estimates**.

# We call the global parameters as **population effects** (or population-level effects, also sometimes called fixed effects)
# and the parameters of each group as **group effects** (or group-level effects, also sometimes called random effects).
# That is why multilevel models are also known as mixed models in which we have both fixed effects and random effects.

# ## Three Approaches to Multilevel Models

# Multilevel models generally fall into three approaches:

# 1. **Random-intercept model**: each group receives a **different intercept** in addition to the global intercept.
# 2. **Random-slope model**: each group receives **different coefficients** for each (or a subset of) independent variable(s) in addition to a global intercept.
# 3. **Random-intercept-slope model**: each group receives **both a different intercept and different coefficients** for each independent variable in addition to a global intercept.

# ### Random-Intercept Model

# The first approach is the **random-intercept model** in which we specify a different intercept for each group,
# in addition to the global intercept. These group-level intercepts are sampled from a hyperprior.

# To illustrate a multilevel model, I will use the linear regression example with a Gaussian/normal likelihood function.
# Mathematically a Bayesian multilevel random-slope linear regression model is:

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Normal}\left( \alpha + \alpha_j + \mathbf{X} \cdot \boldsymbol{\beta}, \sigma \right) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \alpha_j &\sim \text{Normal}(0, \tau) \\
# \boldsymbol{\beta} &\sim \text{Normal}(\mu_{\boldsymbol{\beta}}, \sigma_{\boldsymbol{\beta}}) \\
# \tau &\sim \text{Cauchy}^+(0, \psi_{\alpha})\\
# \sigma &\sim \text{Exponential}(\lambda_\sigma)
# \end{aligned}
# $$

# The priors on the global intercept $\alpha$, global coefficients $\boldsymbol{\beta}$ and error $\sigma$, along with
# the Gaussian/normal likelihood on $\mathbf{y}$ are the same as in the linear regression model.
# But now we have **new parameters**. The first are the **group intercepts** prior $\alpha_j$ that denotes that every group
# $1, 2, \dots, J$ has its own intercept sampled from a normal distribution centered on 0 with a standard deviation $\psi_\alpha$.
# This group intercept is added to the linear predictor inside the Gaussian/normal likelihood function. The **group intercepts' standard
# deviation** $\tau$ have a hyperprior (being a prior of a prior) which is sampled from a positive-constrained Cauchy distribution (a special
# case of the Student-$t$ distribution with degrees of freedom $\nu = 1$) with mean 0 and standard deviation $\sigma_\alpha$.
# This makes the group-level intercept's dispersions being sampled from the same parameter $\tau$ which allows the model
# to use information from one group intercept to infer robust information regarding another group's intercept dispersion and so on.

# This is easily accomplished with Turing:

using Turing
using LinearAlgebra: I
using Statistics: mean, std
using Random: seed!
seed!(123)
setprogress!(false) # hide

@model function varying_intercept(
    X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)
)
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    β ~ filldist(Normal(0, 2), predictors)  # population-level coefficients
    σ ~ Exponential(1 / std(y))             # residual SD
    #prior for variance of random intercepts
    #usually requires thoughtful specification
    τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
    αⱼ ~ filldist(Normal(0, τ), n_gr)       # group-level intercepts

    #likelihood
    ŷ = α .+ X * β .+ αⱼ[idx]
    return y ~ MvNormal(ŷ, σ^2 * I)
end;

# ### Random-Slope Model

# The second approach is the **random-slope model** in which we specify a different slope for each group,
# in addition to the global intercept. These group-level slopes are sampled from a hyperprior.

# To illustrate a multilevel model, I will use the linear regression example with a Gaussian/normal likelihood function.
# Mathematically a Bayesian multilevel random-slope linear regression model is:

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Normal}\left( \alpha + \mathbf{X} \cdot \boldsymbol{\beta}_j \cdot \boldsymbol{\tau}, \sigma \right) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \boldsymbol{\beta}_j &\sim \text{Normal}(0, 1) \\
# \boldsymbol{\tau} &\sim \text{Cauchy}^+(0, \psi_{\boldsymbol{\beta}})\\
# \sigma &\sim \text{Exponential}(\lambda_\sigma)
# \end{aligned}
# $$

# Here we have a similar situation from before with the same hyperprior, but now it is a hyperprior for the the group coefficients'
# standard deviation prior: $\boldsymbol{\beta}_j$.
# This makes the group-level coefficients's dispersions being sampled from the same parameter $\tau$ which allows the model
# to use information from one group coefficients to infer robust information regarding another group's coefficients dispersion and so on.

# In Turing we can accomplish this as:

@model function varying_slope(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2))
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))                    # population-level intercept
    σ ~ Exponential(1 / std(y))                          # residual SD
    #prior for variance of random slopes
    #usually requires thoughtful specification
    τ ~ filldist(truncated(Cauchy(0, 2); lower=0), n_gr) # group-level slopes SDs
    βⱼ ~ filldist(Normal(0, 1), predictors, n_gr)        # group-level standard normal slopes

    #likelihood
    ŷ = α .+ X * βⱼ * τ
    return y ~ MvNormal(ŷ, σ^2 * I)
end;

# ### Random-Intercept-Slope Model

# The third approach is the **random-intercept-slope model** in which we specify a different intercept
# and  slope for each group, in addition to the global intercept.
# These group-level intercepts and slopes are sampled from hyperpriors.

# To illustrate a multilevel model, I will use the linear regression example with a Gaussian/normal likelihood function.
# Mathematically a Bayesian multilevel random-intercept-slope linear regression model is:

# $$
# \begin{aligned}
# \mathbf{y} &\sim \text{Normal}\left( \alpha + \alpha_j + \mathbf{X} \cdot \boldsymbol{\beta}_j \cdot \boldsymbol{\tau}_{\boldsymbol{\beta}}, \sigma \right) \\
# \alpha &\sim \text{Normal}(\mu_\alpha, \sigma_\alpha) \\
# \alpha_j &\sim \text{Normal}(0, \tau_{\alpha}) \\
# \boldsymbol{\beta}_j &\sim \text{Normal}(0, 1) \\
# \tau_{\alpha} &\sim \text{Cauchy}^+(0, \psi_{\alpha})\\
# \boldsymbol{\tau}_{\boldsymbol{\beta}} &\sim \text{Cauchy}^+(0, \psi_{\boldsymbol{\beta}})\\
# \sigma &\sim \text{Exponential}(\lambda_\sigma)
# \end{aligned}
# $$

# Here we have a similar situation from before with the same hyperpriors, but now we fused both random-intercept
# and random-slope together.

# In Turing we can accomplish this as:

@model function varying_intercept_slope(
    X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)
)
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))                     # population-level intercept
    σ ~ Exponential(1 / std(y))                           # residual SD
    #prior for variance of random intercepts and slopes
    #usually requires thoughtful specification
    τₐ ~ truncated(Cauchy(0, 2); lower=0)                 # group-level SDs intercepts
    τᵦ ~ filldist(truncated(Cauchy(0, 2); lower=0), n_gr) # group-level slopes SDs
    αⱼ ~ filldist(Normal(0, τₐ), n_gr)                    # group-level intercepts
    βⱼ ~ filldist(Normal(0, 1), predictors, n_gr)         # group-level standard normal slopes

    #likelihood
    ŷ = α .+ αⱼ[idx] .+ X * βⱼ * τᵦ
    return y ~ MvNormal(ŷ, σ^2 * I)
end;

# In all of the models,
# we are using the `MvNormal` construction where we specify both
# a vector of means (first positional argument)
# and a covariance matrix (second positional argument).
# Regarding the covariance matrix `σ^2 * I`,
# it uses the model's errors `σ`, here parameterized as a standard deviation,
# squares it to produce a variance paramaterization,
# and multiplies by `I`, which is Julia's `LinearAlgebra` standard module implementation
# to represent an identity matrix of any size.

# ## Example - Cheese Ratings

# For our example, I will use a famous dataset called `cheese` (Boatwright, McCulloch & Rossi, 1999), which is data from
# cheese ratings. A group of 10 rural and 10 urban raters rated 4 types of different cheeses (A, B, C and D) in two samples.
# So we have $4 \cdot 20 \cdot2 = 160$ observations and 4 variables:

# * `cheese`: type of cheese from `A` to `D`
# * `rater`: id of the rater from `1` to `10`
# * `background`: type of rater, either `rural` or `urban`
# * `y`: rating of the cheese

# Ok let's read our data with `CSV.jl` and output into a `DataFrame` from `DataFrames.jl`:

using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/cheese.csv"
cheese = CSV.read(HTTP.get(url).body, DataFrame)
describe(cheese)

# As you can see from the `describe()` output, the mean cheese ratings is around 70 ranging from 33 to 91.

# In order to prepare the data for Turing, I will convert the `String`s in variables `cheese` and `background`
# to `Int`s. Regarding `cheese`, I will create 4 dummy variables one for each cheese type; and `background` will be
# converted to integer data taking two values: one for each background type. My intent is to model `background`
# as a group both for intercept and coefficients.
# Take a look at how the data will look like for the first 5 observations:

for c in unique(cheese[:, :cheese])
    cheese[:, "cheese_$c"] = ifelse.(cheese[:, :cheese] .== c, 1, 0)
end

cheese[:, :background_int] = map(cheese[:, :background]) do b
    if b == "rural"
        1
    elseif b == "urban"
        2
    else
        missing
    end
end

first(cheese, 5)

# Now let's us instantiate our model with the data.
# Here, I will specify a vector of `Int`s named `idx` to represent the different observations'
# group memberships. This will be used by Turing when we index a parameter with the `idx`,
# *e.g.* `αⱼ[idx]`.

X = Matrix(select(cheese, Between(:cheese_A, :cheese_D)));
y = cheese[:, :y];
idx = cheese[:, :background_int];

# The first model is the `varying_intercept`:

model_intercept = varying_intercept(X, idx, y)
chain_intercept = sample(model_intercept, NUTS(), MCMCThreads(), 1_000, 4)
println(DataFrame(summarystats(chain_intercept)))

# Here we can see that the model has a population-level intercept `α` along with population-level coefficients `β`s for each `cheese`
# dummy variable. But notice that we have also group-level intercepts for each of the groups `αⱼ`s.
# Specifically, `αⱼ[1]` are the rural raters and `αⱼ[2]` are the urban raters.

# Now let's go to the second model, `varying_slope`:

model_slope = varying_slope(X, idx, y)
chain_slope = sample(model_slope, NUTS(), MCMCThreads(), 1_000, 4)
println(DataFrame(summarystats(chain_slope)))

# Here we can see that the model has still a population-level intercept `α`. But now our population-level
# coefficients `β`s are replaced by group-level coefficients `βⱼ`s along with their standard deviation `τᵦ`s.
# Specifically `βⱼ`'s first index denotes the 4 dummy `cheese` variables' and the second index are the group
# membership. So, for example `βⱼ[1,1]` is the coefficient for `cheese_A` and rural raters (group 1).

# Now let's go to the third model, `varying_intercept_slope`:

model_intercept_slope = varying_intercept_slope(X, idx, y)
chain_intercept_slope = sample(model_intercept_slope, NUTS(), MCMCThreads(), 1_000, 4)
println(DataFrame(summarystats(chain_intercept_slope)))

# Now we have fused the previous model in one. We still have a population-level intercept `α`. But now
# we have in the same model group-level intercepts for each of the groups `αⱼ`s and group-level along with their standard
# deviation `τₐ`. We also have the coefficients `βⱼ`s with their standard deviation `τᵦ`s.
# The parameters are interpreted exactly as the previous cases.

# ## References

# Boatwright, P., McCulloch, R., & Rossi, P. (1999). Account-level modeling for trade promotion: An application of a constrained parameter hierarchical model. Journal of the American Statistical Association, 94(448), 1063–1073.
#
# de Finetti, B. (1974). Theory of Probability (Volume 1). New York: John Wiley & Sons.
#
# Nau, R. F. (2001). De Finetti was Right: Probability Does Not Exist. Theory and Decision, 51(2), 89–124. https://doi.org/10.1023/A:1015525808214
