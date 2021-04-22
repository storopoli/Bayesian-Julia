# # How to use Turing

# [**Turing**](http://turing.ml/) is a ecosystem of Julia packages for Bayesian Inference using
# [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming).
# Turing provide an easy an intuitive way of specifying models.

# ## Probabilistic Programming

# What is a **probabilistic programming** (PP)? It is a **programming paradigm** in which probabilistic models
# are specified and inference for these models is performed **automatically** (Hardesty, 2015). In more clear terms,
# PP and PP Languages (PPLs) allows us to specify **variables as random variables** (like Normal, Binominal etc.) with
# **known or unknown parameters**. Then, we **construct a model** using these variables by specifying how the variables
#  related to each other, and finally **automatic inference of the variables' unknown parameters** is then performed.

# In a Bayesian approach this means specifying **priors**, **likelihoods** and let the PPL compute the **posterior**.
# Since the denominator in the posterior is often intractable, we use Markov Chain Monte Carlo[^MCMC] and some fancy
# algorithm that uses the posterior geometry to guide the MCMC proposal using Hamiltonian dynamics called
# Hamiltonian Monte Carlo (HMC) to approximate the posterior. This involves, besides a suitable PPL, automatic differentiation,
# MCMC chains interface, and also an efficient HMC algorithm implementation. In order to provide all of these features,
# Turing has a whole ecosystem to address each and everyone of these components.

# ## Turing's Ecosystem

# Before we dive into how to specify models in Turing. Let's discuss Turing's ecosystem.
# We have several Julia packages under the Turing's GitHub organization [TuringLang](https://github.com/TuringLang),
# but I will focus on 5 of those:

# * [`Turing.jl`](https://github.com/TuringLang/Turing.jl)
# * [`MCMCChains.jl`](https://github.com/TuringLang/MCMCChains.jl)
# * [`DynamicPPL.jl`](https://github.com/TuringLang/DynamicPPL.jl)
# * [`AdvancedHMC.jl`](https://github.com/TuringLang/AdvancedHMC.jl)
# * [`DistributionsAD.jl`](https://github.com/TuringLang/DistributionsAD.jl)

# The first one is [`Turing.jl`](https://github.com/TuringLang/Turing.jl) itself, the main package that we use to
# **interface with all the Turing ecosystem** of packages and the backbone of the PPL Turing.

# The second, [`MCMCChains.jl`](https://github.com/TuringLang/MCMCChains.jl), is an interface to summarizing MCMC
# simulations and has several utility functions for diagnostics and visualizations.

# The third package is [`DynamicPPL.jl`](https://github.com/TuringLang/DynamicPPL.jl) (Tarek, Xu, Trapp, Ge & Ghahramani, 2020)
# which specifies a domain-specific language and backend for Turing (which itself is a PPL). The main feature of `DynamicPPL.jl`
# is that is is entirely written in Julia and also it is modular.

# [`AdvancedHMC.jl`](https://github.com/TuringLang/AdvancedHMC.jl) provides a robust, modular and efficient implementation
# of advanced HMC algorithms. The state-of-the-art HMC algorithm is the **N**o-**U**-**T**urn **S**ampling
# (NUTS)[^MCMC] (Hoffman & Gelman, 2011) which is available in `AdvancedHMC.jl`.

# Finally, [`DistributionsAD.jl`](https://github.com/TuringLang/DistributionsAD.jl) defines the necessary functions to enable
# automatic differentiation (AD) of the `logpdf` function from [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl)
# using the packages [`Tracker.jl`](https://github.com/FluxML/Tracker.jl), [`Zygote.jl`](https://github.com/FluxML/Zygote.jl),
# [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) and [`ReverseDiff.jl`](https://github.com/JuliaDiff/ReverseDiff.jl).
# The main goal of `DistributionsAD.jl` is to make the output of `logpdf` differentiable with respect to all continuous parameters
# of a distribution as well as the random variable in the case of continuous distributions. This is the package that guarantees the
# "automatical inference" part of the definition of a PPL.

# Most of the time we will not be dealing with neither of these packages directly, since `Turing.jl` will take care of the interfacing
# for us. So let's talk about `Turing.jl`.

# ## `Turing.jl`

# `Turing.jl` is the main package in the Turing ecosystem and the backbone that glues all the other packages together.
# Turing's "workflow" begin with a model specification. We specify the model inside a macro `@model` where we can assign variables
# in two ways:

# * using `~`: which means that a variable follows some probability distribution (Normal, Binomial etc.) and its value is random under that distribution
# * using `=`: which means that a variable does not follow a probability distribution and its value is deterministic (like the normal `=` assignment in programming languages)

# Turing will perform automatic inference on all variables that you specify using `~`.
# Here is a simple example of how we would model a 6-sided dice. Note that a "fair" dice will be distributed as a discrete uniform
# probability with the lower bound as 1 and the upper bound as 6:

# $$ X \sim \text{Uniform}(1,6) \label{uniformdice} $$

# Note that the expectation of $X$ is:

## $$ E(X) = \frac{a+b}{2} = \frac{7}{2} = 3.5 $$

# Graphically this means:

using Plots, StatsPlots, Distributions, LaTeXStrings

dice = DiscreteUniform(1, 6)
plot(dice,
    label="6-sided Dice",
    ms=5,
    xlabel=L"\theta",
    ylabel="Mass",
    ylims=(0, 0.3)
)
vline!([mean(dice)], lw=5, col=:red, label=L"E(\theta)")
savefig(joinpath(@OUTPUT, "dice.svg")); # hide

# So

# \fig{dice}
# \center{*A "fair" 6-sided Dice: Discrete Uniform between 1 and 6*} \\

# ## Footnotes
#
# [^MCMC]: see [5. **Markov Chain Monte Carlo (MCMC)**](/pages/5_MCMC/).

# ## References

# Ge, H., Xu, K., & Ghahramani, Z. (2018). Turing: A Language for Flexible Probabilistic Inference. International Conference on Artificial Intelligence and Statistics, 1682–1690. http://proceedings.mlr.press/v84/ge18b.html
#
# Hardesty (2015).  "Probabilistic programming does in 50 lines of code what used to take thousands". phys.org. April 13, 2015. Retrieved April 13, 2015. https://phys.org/news/2015-04-probabilistic-lines-code-thousands.html
#
# Hoffman, M. D., & Gelman, A. (2011). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593–1623. Retrieved from http://arxiv.org/abs/1111.4246
#
# Tarek, M., Xu, K., Trapp, M., Ge, H., & Ghahramani, Z. (2020). DynamicPPL: Stan-like Speed for Dynamic Probabilistic Models. ArXiv:2002.02702 [Cs, Stat]. http://arxiv.org/abs/2002.02702
