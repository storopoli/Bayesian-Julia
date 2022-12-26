# # Computational Tricks with Turing \\ (Non-Centered Parametrization \\ and QR Decomposition)

# There are some computational tricks that we can employ with Turing.
# I will cover here two computational tricks:

# 1. **QR Decomposition**
# 2. **Non-Centered Parametrization**

# ## QR Decomposition

# Back in "Linear Algebra 101" we've learned that any matrix (even rectangular ones) can be factored
# into the product of two matrices:

# * $\mathbf{Q}$: an orthogonal matrix (its columns are orthogonal unit vectors meaning $\mathbf{Q}^T = \mathbf{Q}^{-1})$.
# * $\mathbf{R}$: an upper triangular matrix.

# This is commonly known as the [**QR Decomposition**](https://en.wikipedia.org/wiki/QR_decomposition):

# $$ \mathbf{A} = \mathbf{Q} \cdot \mathbf{R} $$

# Let me show you an example with a random matrix $\mathbf{A} \in \mathbb{R}^{3 \times 2}$:

A = rand(3, 2)

# Now let's factor `A` using `LinearAlgebra`'s `qr()` function:

using LinearAlgebra: qr, I
Q, R = qr(A)

# Notice that `qr()` produced a tuple containing two matrices `Q` and `R`. `Q` is a 3x3 orthogonal matrix.
# And `R` is a 2x2 upper triangular matrix.
# So that $\mathbf{Q}^T = \mathbf{Q}^{-1}$ (the transpose is equal the inverse):

Matrix(Q') ≈ Matrix(Q^-1)

# Also note that $\mathbf{Q}^T \cdot \mathbf{Q}^{-1} = \mathbf{I}$ (identity matrix):

Q' * Q ≈ I(3)

# This is nice. But what can we do with QR decomposition? It can speed up Turing's sampling by
# a huge factor while also decorrelating the columns of $\mathbf{X}$, *i.e.* the independent variables.
# The orthogonal nature of QR decomposition alters the posterior's topology and makes it easier
# for HMC or other MCMC samplers to explore it. Let's see how fast we can get with QR decomposition.
# First, let's go back to the `kidiq` example in [6. **Bayesian Linear Regression**](/pages/6_linear_reg/):

using Turing
using LinearAlgebra: I
using Statistics: mean, std
using Random: seed!
seed!(123)
setprogress!(false) # hide

@model function linreg(X, y; predictors=size(X, 2))
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))
    β ~ filldist(TDist(3), predictors)
    σ ~ Exponential(1)

    #likelihood
    return y ~ MvNormal(α .+ X * β, σ^2 * I)
end;

using DataFrames
using CSV
using HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/kidiq.csv"
kidiq = CSV.read(HTTP.get(url).body, DataFrame)
X = Matrix(select(kidiq, Not(:kid_score)))
y = kidiq[:, :kid_score]
model = linreg(X, y)
chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)

# See the wall duration in Turing's `chain`: for me it took around 24 seconds.

# Now let's us incorporate QR decomposition in the linear regression model.
# Here, I will use the "thin" instead of the "fat" QR, which scales the $\mathbf{Q}$ and $\mathbf{R}$
# matrices by a factor of $\sqrt{n-1}$ where $n$ is the number of rows of $\mathbf{X}$.
# In practice it is better to implement the thin QR decomposition, which is to be preferred to the fat QR decomposition.
# It is numerically more stable. Mathematically, the thin QR decomposition is:

# $$
# \begin{aligned}
# x &= \mathbf{Q}^* \mathbf{R}^* \\
# \mathbf{Q}^* &= \mathbf{Q} \cdot \sqrt{n - 1} \\
# \mathbf{R}^* &= \frac{1}{\sqrt{n - 1}} \cdot \mathbf{R}\\
# \boldsymbol{\mu}
# &= \alpha + \mathbf{X} \cdot \boldsymbol{\beta} + \sigma
# \\
# &= \alpha + \mathbf{Q}^* \cdot \mathbf{R}^* \cdot \boldsymbol{\beta} + \sigma
# \\
# &= \alpha + \mathbf{Q}^* \cdot (\mathbf{R}^* \cdot \boldsymbol{\beta}) + \sigma
# \\
# &= \alpha + \mathbf{Q}^* \cdot \widetilde{\boldsymbol{\beta}} + \sigma
# \\
# \end{aligned}
# $$

# Then we can recover the original $\boldsymbol{\beta}$ with:

# $$ \boldsymbol{\beta} = \mathbf{R}^{*-1} \cdot \widetilde{\boldsymbol{\beta}} $$

# In Turing, a model with QR decomposition would be the same `linreg` but with a
# different `X` matrix supplied, since it is a data transformation. First, we
# decompose your model data `X` into `Q` and `R`:

Q, R = qr(X)
Q_ast = Matrix(Q) * sqrt(size(X, 1) - 1)
R_ast = R / sqrt(size(X, 1) - 1);

# Then, we instantiate a model with `Q` instead of `X` and sample as you would:

model_qr = linreg(Q_ast, y)
chain_qr = sample(model_qr, NUTS(1_000, 0.65), MCMCThreads(), 1_000, 4)

# See the wall duration in Turing's `chain_qr`: for me it took around 5 seconds. Much faster than
# the regular `linreg`.
# Now we have to reconstruct our $\boldsymbol{\beta}$s:

betas = mapslices(
    x -> R_ast^-1 * x, chain_qr[:, namesingroup(chain_qr, :β), :].value.data; dims=[2]
)
chain_beta = setrange(
    Chains(betas, ["real_β[$i]" for i in 1:size(Q_ast, 2)]), 1_001:1:3_000
)
chain_qr_reconstructed = hcat(chain_beta, chain_qr)

# ## Non-Centered Parametrization

# Now let's us explore **Non-Centered Parametrization** (NCP). This is useful when the posterior's
# topology is very difficult to explore as has regions where HMC sampler has to
# change the step size $L$ and the $\epsilon$ factor. This is  I've showed one of the most infamous
# case in [5. **Markov Chain Monte Carlo (MCMC)**](/pages/5_MCMC/): Neal's Funnel (Neal, 2003):

using CairoMakie
using Distributions
funnel_y = rand(Normal(0, 3), 10_000)
funnel_x = rand(Normal(), 10_000) .* exp.(funnel_y / 2)

f, ax, s = scatter(
    funnel_x,
    funnel_y;
    color=(:steelblue, 0.3),
    axis=(; xlabel=L"X", ylabel=L"Y", limits=(-100, 100, nothing, nothing)),
)
save(joinpath(@OUTPUT, "funnel.svg"), f); # hide

# \fig{funnel}
# \center{*Neal's Funnel*} \\

# Here we see that in upper part of the funnel HMC has to take few steps $L$ and be more liberal with
# the $\epsilon$ factor. But, the opposite is in the lower part of the funnel: way more steps $L$ and be
# more conservative with the $\epsilon$ factor.

# To see the devil's funnel (how it is known in some Bayesian circles) in action, let's code it in Turing and then sample:

@model function funnel()
    y ~ Normal(0, 3)
    return x ~ Normal(0, exp(y / 2))
end

chain_funnel = sample(funnel(), NUTS(), MCMCThreads(), 1_000, 4)

# Wow, take a look at those `rhat` values... That sucks: all are above `1.01` even with 4 parallel chains with 1,000
# iterations!

# How do we deal with that? We **reparametrize**! Note that we can add two normal distributions in the following manner:

# $$ \text{Normal}(\mu, \sigma) = \text{Standard Normal} \cdot \sigma + \mu $$

# where the standard normal is the normal with mean $\mu = 0$ and standard deviation $\sigma = 1$.
# This is why is called Non-Centered Parametrization because we "decouple" the parameters and
# reconstruct them before.

@model function ncp_funnel()
    x̃ ~ Normal()
    ỹ ~ Normal()
    y = 3.0 * ỹ         # implies y ~ Normal(0, 3)
    return x = exp(y / 2) * x̃  # implies x ~ Normal(0, exp(y / 2))
end

chain_ncp_funnel = sample(ncp_funnel(), NUTS(), MCMCThreads(), 1_000, 4)

# Much better now: all `rhat` are well below `1.01` (or below `0.99`).

# How we would implement this a real-world model in Turing? Let's go back to the `cheese` random-intercept model
# in [10. **Multilevel Models (a.k.a. Hierarchical Models)**](/pages/10_multilevel_models/). Here was the
# approach that we took, also known as Centered Parametrization (CP):

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
    αⱼ ~ filldist(Normal(0, τ), n_gr)       # CP group-level intercepts

    #likelihood
    ŷ = α .+ X * β .+ αⱼ[idx]
    return y ~ MvNormal(ŷ, σ^2 * I)
end;

# To perform a Non-Centered Parametrization (NCP) in this model we do as following:

@model function varying_intercept_ncp(
    X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2)
)
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    β ~ filldist(Normal(0, 2), predictors)  # population-level coefficients
    σ ~ Exponential(1 / std(y))             # residual SD

    #prior for variance of random intercepts
    #usually requires thoughtful specification
    τ ~ truncated(Cauchy(0, 2); lower=0)   # group-level SDs intercepts
    zⱼ ~ filldist(Normal(0, 1), n_gr)      # NCP group-level intercepts

    #likelihood
    ŷ = α .+ X * β .+ zⱼ[idx] .* τ
    return y ~ MvNormal(ŷ, σ^2 * I)
end;

# Here we are using a NCP with the `zⱼ`s following a standard normal and we reconstruct the
# group-level intercepts by multiplying the `zⱼ`s by `τ`. Since the original `αⱼ`s had a prior
# centered on 0 with standard deviation `τ`, we only have to use the multiplication by `τ`
# to get back the `αⱼ`s.

# Now let's see how NCP compares to the CP. First, let's redo our CP hierarchical model:

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/cheese.csv"
cheese = CSV.read(HTTP.get(url).body, DataFrame)

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

X = Matrix(select(cheese, Between(:cheese_A, :cheese_D)));
y = cheese[:, :y];
idx = cheese[:, :background_int];

model_cp = varying_intercept(X, idx, y)
chain_cp = sample(model_cp, NUTS(), MCMCThreads(), 1_000, 4)

# Now let's do the NCP hierarchical model:

model_ncp = varying_intercept_ncp(X, idx, y)
chain_ncp = sample(model_ncp, NUTS(), MCMCThreads(), 1_000, 4)

# Notice that some models are better off with a standard Centered Parametrization (as is our `cheese` case here).
# While others are better off with a Non-Centered Parametrization. But now you know how to apply both parametrizations
# in Turing. Before we conclude, we need to recover our original `αⱼ`s. We can do this by multiplying `zⱼ[idx] .* τ`:

τ = summarystats(chain_ncp)[:τ, :mean]
αⱼ = mapslices(
    x -> x * τ, chain_ncp[:, namesingroup(chain_ncp, :zⱼ), :].value.data; dims=[2]
)
chain_ncp_reconstructed = hcat(
    MCMCChains.resetrange(chain_ncp), Chains(αⱼ, ["αⱼ[$i]" for i in 1:length(unique(idx))])
)

# ## References

# Neal, Radford M. (2003). Slice Sampling. The Annals of Statistics, 31(3), 705–741. Retrieved from https://www.jstor.org/stable/3448413
