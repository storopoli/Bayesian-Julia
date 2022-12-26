# # Common Probability Distributions

# Bayesian statistics uses probability distributions as the inference "engine" for the estimation of the parameter values
# along with their uncertainties.

# Imagine that probability distributions are small pieces of "Lego". We can build whatever we want with these little pieces.
# We can make a castle, a house, a city; literally anything we want. The same is true for probabilistic models in Bayesian
# statistics. We can build models from the simplest to the most complex using probability distributions and their relationships
# to each other. In this tutorial we will give a brief overview of the main probabilistic distributions, their mathematical
# notation and their main uses in Bayesian statistics.

# A probability distribution is the mathematical function that gives the probabilities of occurrence
# of different possible outcomes for an experiment. It is a mathematical description of a random phenomenon in terms of its
# sample space and the probabilities of events (subsets of the sample space).

# We generally use the notation `X ~ Dist (par1, par2, ...)`. Where `X` is the variable,` Dist` is the name of the distribution,
# and `par` are the parameters that define how the distribution behaves. Any probabilistic distribution can be "parameterized"
# by specifying parameters that allow us to shape some aspects of the distribution for some specific purpose.

# Let's start with discrete distributions and then we'll address the continuous ones.

# ## Discrete

# Discrete probability distributions are those where the results are discrete numbers (also called whole numbers):
# $\dots, -2, 1, 0, 1, 2, \dots, N$ and $N \in \mathbb{Z}$. In discrete distributions we say the probability that
# a distribution takes certain values as "mass". The probability mass function $\text {PMF}$ is the function that
# specifies the probability of the random variable $X$ taking the value $x$:

# $$ \text{PMF}(x) = P(X = x) $$

# ### Discrete Uniform

# The discrete uniform distribution is a symmetric probability distribution in which a finite number of values are equally
# likely to be observed. Each of the $n$ values has an equal probability $\frac{1}{n}$. Another way of saying
# "discrete uniform distribution" would be "a known and finite number of results equally likely to happen".

# The discrete uniform distribution has two parameters and its notation is $\text{Unif} (a, b)$:

# * Lower Bound ($a$)
# * Upper Bound ($b$)

# Example: a 6-sided dice.

using CairoMakie
using Distributions

f, ax, b = barplot(
    DiscreteUniform(1, 6);
    axis=(;
        title="6-sided Dice",
        xlabel=L"\theta",
        ylabel="Mass",
        xticks=1:6,
        limits=(nothing, nothing, 0, 0.3),
    ),
)
save(joinpath(@OUTPUT, "discrete_uniform.svg"), f); # hide

# \fig{discrete_uniform}
# \center{*Discrete Uniform between 1 and 6*} \\

# ### Bernoulli

# Bernoulli's distribution describes a binary event of a successful experiment. We usually represent $0$ as failure and $1$
# as success, so the result of a Bernoulli distribution is a binary variable $Y \in \{ 0, 1 \}$.

# The Bernoulli distribution is widely used to model discrete binary outcomes in which there are only two possible results.

# Bernoulli's distribution has only a single parameter and its notation is $\text{Bernoulli}(p)$:

# * Success Probability ($p$)

# Example: Whether the patient survived or died or whether the customer completes their purchase or not.

f, ax1, b = barplot(
    Bernoulli(0.5);
    width=0.3,
    axis=(;
        title=L"p=0.5",
        xlabel=L"\theta",
        ylabel="Mass",
        xticks=0:1,
        limits=(nothing, nothing, 0, 1),
    ),
)
ax2 = Axis(
    f[1, 2]; title=L"p=0.2", xlabel=L"\theta", xticks=0:1, limits=(nothing, nothing, 0, 1)
)
barplot!(ax2, Bernoulli(0.2); width=0.3)
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "bernoulli.svg"), f); # hide

# \fig{bernoulli}
# \center{*Bernoulli with $p = \{ 0.5, 0.2 \}$*} \\

# ### Binomial

# The binomial distribution describes an event of the number of successes in a sequence of $n$ **independent** experiment(s),
# each asking a yes-no question with a probability of success $p$. Note that the Bernoulli distribution is a special case
# of the binomial distribution where the number of experiments is $1$.

# The binomial distribution has two parameters and its notation is $\text{Bin} (n, p)$ or $ \text{Binomial} (n, p)$:

# * Number of Experiment(s) ($n$)
# * Probability of Success ($p$)

# Example: number of heads in 5 coin flips.

f, ax1, b = barplot(
    Binomial(5, 0.5); axis=(; title=L"p=0.5", xlabel=L"\theta", ylabel="Mass")
)
ax2 = Axis(f[1, 2]; title=L"p=0.2", xlabel=L"\theta")
barplot!(ax2, Binomial(5, 0.2))
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "binomial.svg"), f); # hide
# \fig{binomial}
# \center{*Binomial with $n=5$ and $p = \{ 0.5, 0.2 \}$*} \\

# ### Poisson

# The Poisson distribution expresses the probability that a given number of events will occur in a fixed interval of time
# or space if those events occur with a known constant average rate and regardless of the time since the last event. The
# Poisson distribution can also be used for the number of events at other specified intervals, such as distance, area or volume.

# The Poisson distribution has one parameter and its notation is $\text{Poisson} (\lambda)$:

# * Rate ($\lambda$)

# Example: Number of emails you receive daily. Number of holes you find on the street.

f, ax1, b = barplot(
    Poisson(1); axis=(; title=L"\lambda=1", xlabel=L"\theta", ylabel="Mass")
)
ax2 = Axis(f[1, 2]; title=L"\lambda=4", xlabel=L"\theta")
barplot!(ax2, Poisson(4))
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "poisson.svg"), f); # hide

# \fig{poisson}
# \center{*Poisson with $\lambda = \{ 1, 4 \}$*} \\

# ### Negative Binomial

# The negative binomial distribution describes an event of the number of failures before the $k$th success in a sequence of $n$ independent experiment(s),
# each asking a yes-no question with probability $p$.
# Note that it becomes identical to the Poisson distribution at the limit of $k \to \infty$.
# This makes the negative binomial a robust option to replace a Poisson
# distribution to model phenomena with a overdispersion* (excess expected variation in data).

# The negative binomial distribution has two parameters and its notation is $\text{NB} (k, p)$ or $\text{Negative-Binomial} (k, p)$:

# * Number of Success(es) ($k$)
# * Probability of Success ($p$)

# Any phenomenon that can be modeled with a Poisson distribution, can be modeled with a negative binomial distribution
# (Gelman et al., 2013; 2020).

# Example: Annual count of tropical cyclones.

f, ax1, b = barplot(
    NegativeBinomial(1, 0.5); axis=(; title=L"k=1", xlabel=L"\theta", ylabel="Mass")
)
ax2 = Axis(f[1, 2]; title=L"k=2", xlabel=L"\theta")
barplot!(ax2, NegativeBinomial(2, 0.5))
linkaxes!(ax1, ax2)
save(joinpath(@OUTPUT, "negbinomial.svg"), f); # hide

# \fig{negbinomial}
# \center{*Negative Binomial with $p=0.5$ and $r = \{ 1, 2 \}$*} \\

# ## Continuous

# Continuous probability distributions are those where the results are values in a continuous range (also called real numbers):
# $(-\infty, +\infty) \in \mathbb{R}$. In continuous distributions we call the probability that a distribution takes certain
# values as "density". As we are talking about real numbers we are not able to obtain the probability that a random variable
# $X$ takes the value of $x$. This will always be $0$, as there is no way to specify an exact value of $x$. $x$ lives in the
# real numbers line, so we need to specify the probability that $X$ takes values in a **range** $[a,b]$. The probability
# density function $\text {PDF}$ is defined as:

# $$ \text{PDF}(x) = P(a \leq X \leq b) = \int_a^b f(x) dx $$

# ### Normal / Gaussian

# This distribution is generally used in the social and natural sciences to represent continuous variables in which its
# distributions are not known. This assumption is due to the central limit theorem. The central limit theorem states that,
# in some conditions, the average of many samples (observations) of a random variable with finite mean and variance is
# itself a random variable whose distribution converges to a normal distribution as the number of samples increases.
# Therefore, physical quantities that are expected to be the sum of many independent processes (such as measurement errors)
# often have distributions that are expected to be nearly normal.

# The normal distribution has two parameters and its notation is $\text{Normal} (\mu, \sigma^2)$ or $\text{N}(\mu, \sigma^2)$:

# * Mean ($\mu$): distribution mean which is also both the mode and the median of the distribution
# * Standard Deviation ($\sigma$): the variance of the distribution ($\sigma^2$) is a measure of the dispersion of the observations in relation to the mean

# Example: Height, Weight, etc.

f, ax, l = lines(
    Normal(0, 1);
    label=L"\sigma=1",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(-4, 4, nothing, nothing)),
)
lines!(ax, Normal(0, 0.5); label=L"\sigma=0.5", linewidth=5)
lines!(ax, Normal(0, 2); label=L"\sigma=2", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "normal.svg"), f); # hide

# \fig{normal}
# \center{*Normal with $\mu=0$ and $\sigma = \{ 1, 0.5, 2 \}$*} \\

# ### Log-normal

# The Log-normal distribution is a continuous probability distribution of a random variable whose logarithm is normally
# distributed. Thus, if a random variable $X$ is normally distributed by its natural log, then $Y =\log(X)$ will have
# a normal distribution.

# A random variable with logarithmic distribution accepts only positive real values. It is a convenient and useful model
# for measurements in the physical sciences and engineering, as well as medicine, economics and other fields,
# eg. for energies, concentrations, lengths, financial returns and other values.

# A log-normal process is the statistical realization of the multiplicative product of many independent random variables,
# each one being positive.

# The log-normal distribution has two parameters and its notation is $\text{Log-Normal} (\mu, \sigma^2)$:

# * Mean ($\mu$): natural logarithm of the mean the distribution
# * Standard Deviation ($\sigma$): natural logarithm of the variance of the distribution ($\sigma^2$) is a measure of the dispersion of the observations in relation to the mean

f, ax, l = lines(
    LogNormal(0, 1);
    label=L"\sigma=1",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(0, 3, nothing, nothing)),
)
lines!(ax, LogNormal(0, 0.25); label=L"\sigma=0.25", linewidth=5)
lines!(ax, LogNormal(0, 0.5); label=L"\sigma=0.5", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "lognormal.svg"), f); # hide

# \fig{lognormal}
# \center{*Log-Normal with $\mu=0$ and $\sigma = \{ 1, 0.25, 0.5 \}$*} \\

# ### Exponential

# The exponential distribution is the probability distribution of time between events
# that occur continuously and independently at a constant average rate.

# The exponential distribution has one parameter and its notation is $\text{Exp} (\lambda)$:

# * Rate ($\lambda$)

# Example: How long until the next earthquake. How long until the next bus arrives.

f, ax, l = lines(
    Exponential(1);
    label=L"\lambda=1",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(0, 4.5, nothing, nothing)),
)
lines!(ax, Exponential(0.5); label=L"\lambda=0.5", linewidth=5)
lines!(ax, Exponential(1.5); label=L"\lambda=2", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "exponential.svg"), f); # hide

# \fig{exponential}
# \center{*Exponential with $\lambda = \{ 1, 0.5, 1.5 \}$*} \\

# ### Student-$t$ distribution

# Student-$t$ distribution appears when estimating the average of a population normally distributed in situations
# where the sample size is small and the population standard deviation is unknown.

# If we take a sample of $n$ observations from a normal distribution, then the distribution
# Student-$t$ with $\nu = n-1$ degrees of freedom can be defined as the distribution of the location of the
# sample mean relative to the true mean, divided by the standard deviation of the sample, after multiplying by
# the standardizing term $\sqrt{n}$.

# The Student-$t$ distribution is symmetrical and bell-shaped, like the normal distribution, but has longer tails,
# which means that it is more likely to produce values ​​that are far from its mean.

# The Student-$t$ distribution has one parameter and its notation is $\text{Student-$t$} (\nu)$:

# * Degrees of Freedom ($\nu$): controls how much it resembles a normal distribution

# Example: A database full of outliers.

f, ax, l = lines(
    TDist(2);
    label=L"\nu=2",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(-4, 4, nothing, nothing)),
)
lines!(ax, TDist(8); label=L"\nu=8", linewidth=5)
lines!(ax, TDist(30); label=L"\nu=30", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "tdist.svg"), f); # hide

# \fig{tdist}
# \center{*Student-$t$ with $\nu = \{ 2, 8, 30 \}$*} \\

# ### Beta Distribution

# The beta distributions is a natural choice to model anything that is constrained to take values between 0 and 1.
# So it is a good candidate for probabilities and proportions.

# The beta distribution has two parameters and its notation is $\text{Beta} (a, b)$:

# * Shape parameter ($a$ or sometimes $\alpha$): controls how much the shape is shifted towards 1
# * Shape parameter ($b$ or sometimes $\beta$): controls how much the shape is shifted towards 0

# Example: A basketball player has made already scored 5 free throws while missing 3 in a total of 8 attempts
# -- $\text{Beta}(3, 5)$.

f, ax, l = lines(
    Beta(1, 1);
    label=L"a=b=1",
    linewidth=5,
    axis=(; xlabel=L"\theta", ylabel="Density", limits=(0, 1, nothing, nothing)),
)
lines!(ax, Beta(3, 2); label=L"a=3, b=2", linewidth=5)
lines!(ax, Beta(2, 3); label=L"a=2, b=3", linewidth=5)
axislegend(ax)
save(joinpath(@OUTPUT, "beta.svg"), f); # hide

# \fig{beta}
# \center{*Beta with different values of $a$ and $b$*} \\

# ## Distribution Zoo

# I did not cover all existing distributions. There is a whole plethora of probabilistic distributions.

# To access the entire "distribution zoo" use this tool from [Ben Lambert](https://ben-lambert.com/bayesian/)
# (statistician from *Imperial College of London*): <https://ben18785.shinyapps.io/distribution-zoo/>

# ## References

# Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis. Chapman and Hall/CRC.
#
# Gelman, A., Hill, J., & Vehtari, A. (2020). Regression and other stories. Cambridge University Press.
