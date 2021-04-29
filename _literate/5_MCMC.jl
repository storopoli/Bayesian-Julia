# # Markov Chain Monte Carlo (MCMC)

# The main computational barrier for Bayesian statistics is the denominator $P(\text{data})$ of the Bayes formula:

# $$ P(\theta \mid \text{data})=\frac{P(\theta) \cdot P(\text{data} \mid \theta)}{P(\text{data})} \label{bayes} $$

# In discrete cases we can turn the denominator into a sum of all parameters using the chain rule of probability:

# $$ P(A,B \mid C)=P(A \mid B,C) \times P(B \mid C) \label{chainrule} $$

# This is also called marginalization:

# $$ P(\text{data})=\sum_{\theta} P(\text{data} \mid \theta) \times P(\theta) \label{discretemarginalization} $$

# However, in the continuous cases the denominator $P(\text{data})$ becomes a very large and complicated integral to calculate:

# $$ P(\text{data})=\int_{\theta} P(\text{data} \mid \theta) \times P(\theta)d \theta \label{continuousmarginalization} $$

# In many cases this integral becomes *intractable* (incalculable) and therefore we must find other ways to calculate
# the posterior probability $P(\theta \mid \text{data})$ in \eqref{bayes} without using the denominator $P(\text{data})$.

# ## What is the denominator $P(\text{data})$ for?

# Quick answer: to normalize the posterior in order to make it a valid probability distribution. This means that the sum of all probabilities
# of the possible events in the probability distribution must be equal to 1:

# - in the case of discrete probability distribution:  $\sum_{\theta} P(\theta \mid \text{data}) = 1$
# - in the case of continuous probability distribution: $\int_{\theta} P(\theta \mid \text{data})d \theta = 1$

# ## What if we remove this denominator?

# When we remove the denominator $(\text{data})$ we have that the posterior $P(\theta \mid \text{data})$ is **proportional** to the
# prior multiplied by the likelihood $P(\theta) \cdot P(\text{data} \mid \theta)$[^propto].

# $$ P(\theta \mid \text{data}) \propto P(\theta) \cdot P(\text{data} \mid \theta) \label{proptobayes} $$

# This [YouTube video](https://youtu.be/8FbqSVFzmoY) explains the denominator problem very well, see below:

# ~~~
# <style>.embed-container { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; } .embed-container iframe, .embed-container object, .embed-container embed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }</style><div class='embed-container'><iframe src='https://www.youtube.com/embed/8FbqSVFzmoY' frameborder='0' allowfullscreen></iframe></div>
# ~~~

# ## Markov Chain Monte Carlo (MCMC)

# This is where Markov Chain Monte Carlo comes in. MCMC is a broad class of computational tools for approximating integrals and generating samples from
# a posterior probability (Brooks, Gelman, Jones & Meng, 2011). MCMC is used when it is not possible to sample $\theta$ directly from the subsequent
# probabilistic distribution $P(\theta \mid \text{data})$. Instead, we sample in an iterative manner such that at each step of the process we expect the
# distribution from which we sample $P^* (\theta^* \mid \text{data})$ (here $*$ means simulated) becomes increasingly similar to the posterior
# $P(\theta \mid \text{data})$. All of this is to eliminate the (often impossible) calculation of the denominator $P(\text{data})$.

# The idea is to define an ergodic Markov chain (that is to say that there is a single stationary distribution) of which the set of possible states
# is the sample space and the stationary distribution is the distribution to be approximated (or sampled). Let $X_0, X_1, \dots, X_n$ be a
# simulation of the chain. The Markov chain converges to the stationary distribution of any initial state $X_0$ after a large enough number
# of iterations $r$, the distribution of the state $X_r$ will be similar to the stationary distribution, so we can use it as a sample.
# Markov chains have a property that the probability distribution of the next state depends only on the current state and not on the
# sequence of events that preceded: $P(X_{n+1}=x \mid X_{0},X_{1},X_{2},\ldots ,X_{n}) = P(X_{n+1}=x \mid X_{n})$. This property is
# called Markovian, after the mathematician [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov) (see figure below).
# Similarly, repeating this argument with $X_r$ as the starting point, we can use $X_{2r}$ as a sample, and so on.
# We can then use the state sequence $X_r, X_{2r}, X_{3r}, \dots$ as almost independent samples of the stationary distribution
# of the Markov chain.

# ![Andrey Markov](/pages/images/andrey_markov.jpg)
#
# \center{*Andrey Markov*} \\

# The effectiveness of this approach depends on:

# 1. how big $r$ must be to ensure a suitably good sample; and

# 2. computational power required for each iteration of the Markov chain.

# In addition, it is customary to discard the first iterations of the algorithm as they are usually not representative
# of the distribution to be approximated. In the initial iterations of MCMC algorithms, generally the Markov current is
# in a *warm-up* process[^warmup] and its state is far from ideal to start a reliable sampling. It is generally recommended
# to discard half of the iterations (Gelman, Carlin, Stern, Dunson, Vehtari, & Rubin, 2013a). For example:
# if the Markov chain has 4,000 iterations, we discard the first 2,000 as warm-up.

# ### Monte Carlo Method

# Stanislaw Ulam (figure below), who participated in the Manhattan project and when trying to calculate the neutron diffusion
# process for the hydrogen bomb ended up creating a class of methods called **_Monte Carlo_**.

# ![Stanislaw Ulam](/pages/images/stanislaw.jpg)
#
# \center{*Stanislaw Ulam*} \\

# Monte Carlo methods have the underlying concept of using randomness to solve problems that can be deterministic in principle.
# They are often used in physical and mathematical problems and are most useful when it is difficult or impossible to use other approaches.
# Monte Carlo methods are used mainly in three classes of problems: optimization, numerical integration and generating sample from a
# probability distribution.

# The idea for the method came to Ulam while playing solitaire during his recovery from surgery, as he thought about playing hundreds
# of games to statistically estimate the probability of a successful outcome. As he himself mentions in Eckhardt (1987):
#
# > "The first thoughts and attempts I made to practice [the Monte Carlo method] were suggested by a question which occurred to me
# > in 1946 as I was convalescing from an illness and playing solitaires. The question was what are the chances that a Canfield solitaire
# > laid out with 52 cards will come out successfully? After spending a lot of time trying to estimate them by pure combinatorial
# > calculations, I wondered whether a more practical method than "abstract thinking" might not be to lay it out say one hundred times and
# > simply observe and count the number of successful plays. This was already possible to envisage with the beginning of the new era of
# > fast computers, and I immediately thought of problems of neutron diffusion and other questions of mathematical physics, and more
# > generally how to change processes described by certain differential equations into an equivalent form interpretable as a succession
# > of random operations. Later... [in 1946, I ] described the idea to John von Neumann and we began to plan actual calculations."

# Because it was secret, von Neumann and Ulam's work required a codename. A colleague of von Neumann and Ulam, Nicholas Metropolis
# (figure below), suggested using the name "Monte Carlo", which refers to Casino Monte Carlo in Monaco, where Ulam's uncle
# (Michał Ulam) borrowed money from relatives to gamble.

# ![Nicholas Metropolis](/pages/images/nicholas_metropolis.png)
#
# \center{*Nicholas Metropolis*} \\

# The [applications of the Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method#Applications) are numerous:
# physical sciences, engineering, climate change, computational biology, computer graphics, applied statistics, artificial intelligence,
# search and rescue, finance and business and law. In the scope of these tutorials we will focus on applied statistics and specifically
# in the context of Bayesian inference: providing a random sample of the posterior distribution.

# ### Simulations

# I will do some simulations to ilustrate MCMC algorithms and techniques. So, here's the initial setup:

using Plots, StatsPlots, Distributions, LaTeXStrings, Random

Random.seed!(123);

# Let's start with a toy problem of a multivariate normal distribution of $X$ and $Y$, where

# $$
# \begin{bmatrix}
# X \\
# Y
# \end{bmatrix} \sim \text{Multivariate Normal} \left(
# \begin{bmatrix}
# \mu_X \\
# \mu_Y
# \end{bmatrix}, \mathbf{\Sigma}
# \right) \\
# \mathbf{\Sigma} \sim
# \begin{pmatrix}
# \sigma^2_{X} & \sigma_{X}\sigma_{Y} \rho \\
# \sigma_{X}\sigma_{Y} \rho & \sigma^2_{Y}
# \end{pmatrix}
# \label{mvnormal}
# $$

# If we assign $\mu_X = \mu_Y = 0$ and $\sigma_X = \sigma_Y = 1$ (mean 0 and standard deviation 1
# for both $X$ and $Y$), we have the following formulation from \eqref{mvnormal}:

# $$
# \begin{bmatrix}
# X \\
# Y
# \end{bmatrix} \sim \text{Multivariate Normal} \left(
# \begin{bmatrix}
# 0 \\
# 0
# \end{bmatrix}, \mathbf{\Sigma}
# \right), \\
# \mathbf{\Sigma} \sim
# \begin{pmatrix}
# 1 & \rho \\
# \rho & 1
# \end{pmatrix}
# \label{stdmvnormal}
# $$

# All that remains is to assign a value for $\rho$ in \eqref{stdmvnormal} for the correlation between $X$ and $Y$.
# For our example we will use correlation of 0.8 ($\rho = 0.8$):

# $$
# \mathbf{\Sigma} \sim
# \begin{pmatrix}
# 1 & 0.8 \\
# 0.8 & 1
# \end{pmatrix}
# \label{Sigma}
# $$

const N = 100_000
const μ = [0, 0]
const Σ = [1 0.8; 0.8 1]

const mvnormal = MvNormal(μ, Σ)

data = rand(mvnormal, N)';

# In the figure below it is possible to see a countour plot of the PDF of a multivariate normal distribution composed of two normal
# variables $X$ and $Y$, both with mean 0 and standard deviation 1.
# The correlation between $X$ and $Y$ is $\rho = 0.8$:

x = -3:0.01:3
y = -3:0.01:3
dens_mvnormal = [pdf(mvnormal, [i, j]) for i in x, j in y]
contour(x, y, dens_mvnormal, xlabel=L"X", ylabel=L"Y", fill=true)
savefig(joinpath(@OUTPUT, "countour_mvnormal.svg")); # hide

# \fig{countour_mvnormal}
# \center{*Countour Plot of the PDF of a Multivariate Normal Distribution*} \\

# Also a surface plot can be seen below for you to get a 3-D intuition of what is going on:

surface(x, y, dens_mvnormal, xlabel=L"X", ylabel=L"Y", zlabel="PDF")
savefig(joinpath(@OUTPUT, "surface_mvnormal.svg")); # hide

# \fig{surface_mvnormal}
# \center{*Surface Plot of the PDF of a Multivariate Normal Distribution*} \\

# ### Metropolis and Metropolis-Hastings

# The first MCMC algorithm widely used to generate samples from Markov chain originated in physics in the 1950s
# (in a very close relationship with the atomic bomb at the Manhattan project) and is called **Metropolis**
# (Metropolis, Rosenbluth, Rosenbluth, Teller, & Teller, 1953) in honor of the first author [Nicholas Metropolis](https://en.wikipedia.org/wiki/Nicholas_Metropolis)
# (figure above). In summary, the Metropolis algorithm is an adaptation of a random walk with
# an acceptance/rejection rule to converge to the target distribution.

# The Metropolis algorithm uses a **proposal distribution** $J_t(\theta^*)$ ($J$ stands for *jumping distribution*
# and $t$ indicates which state of the Markov chain we are in) to define next values of the distribution
# $P^*(\theta^* \mid \text{data})$. This distribution must be symmetrical:

# $$ J_t (\theta^* \mid \theta^{t-1}) = J_t(\theta^{t-1} \mid \theta^*) \label{symjump} $$

# In the 1970s, a generalization of the Metropolis algorithm emerged that **does not** require that the proposal distributions
# be symmetric. The generalization was proposed by [Wilfred Keith Hastings](https://en.wikipedia.org/wiki/W._K._Hastings)
# (Hastings, 1970) (figure below) and is called **Metropolis-Hastings algorithm**.

# ![Wilfred Hastings](/pages/images/hastings.jpg)
#
# \center{*Wilfred Hastings*} \\

# #### Metropolis Algorithm

# The essence of the algorithm is a random walk through the parameters' sample space, where the probability of the Markov chain
# changing state is defined as:

# $$ P_{\text{change}} = \min \left( {\frac{P (\theta_{\text{proposed}})}{P (\theta_{\text{current}})}}, 1 \right) \label{proposal} $$

# This means that the Markov chain will only change to a new state under two conditions:

# 1. When the probability of the parameters proposed by the random walk $P(\theta_{\text{proposed}})$ is **greater** than the probability of the parameters of the current state $P(\theta_{\text{current}})$, we change with 100% probability. Note that if $P(\theta_{\text{proposed}}) > P(\theta_{\text{current}})$ then the function $\min$ chooses the value 1 which means 100%.
# 2. When the probability of the parameters proposed by the random walk $P(\theta_{\text{proposed}})$ is **less** than the probability of the parameters of the current state $P(\theta_{\text{current}})$, we changed with a probability equal to the proportion of that difference. Note that if $P(\theta_{\text{proposed}}) < P(\theta_{\text{current}})$ then the function $\min$ **does not** choose the value 1, but the value $\frac{P(\theta_{\text{proposed}})}{P(\theta_{\text{current}})}$ which equates the proportion of the probability of the proposed parameters to the probability of the parameters at the current state.

# Anyway, at each iteration of the Metropolis algorithm, even if the Markov chain changes state or not, we sample the parameter
# $\theta$ anyway. That is, if the chain does not change to a new state, $\theta$ will be sampled twice (or
# more if the current is stationary in the same state).

# The Metropolis-Hastings algorithm can be described in the following way [^metropolis] ($\theta$ is the parameter, or set of
# parameters, of interest and $y$ is the data):

# 1. Define a starting point $\theta^0$ of which $p(\theta^0 \mid y) > 0$, or sample it from an initial distribution $p_0(\theta)$. $p_0(\theta)$ can be a normal distribution or a prior distribution of $\theta$ ($p(\theta)$).
#
# 2. For $t = 1, 2, \dots$:

#    -   Sample a proposed $\theta^*$ from a proposal distribution in time $t$, $J_t (\theta^* \mid \theta^{t-1})$.

#    -   Calculate the ratio of probabilities:

#        -   **Metropolis**: $r = \frac{p(\theta^*  \mid y)}{p(\theta^{t-1} \mid y)}$
#        -   **Metropolis-Hastings**: $r = \frac{\frac{p(\theta^* \mid y)}{J_t(\theta^* \mid \theta^{t-1})}}{\frac{p(\theta^{t-1} \mid y)}{J_t(\theta^{t-1} \mid \theta^*)}}$

#    -   Assign:

#        $
#        \theta^t =
#        \begin{cases}
#        \theta^* & \text{with probability } \min (r, 1) \\
#        \theta^{t-1} & \text{otherwise}
#        \end{cases}
#        $

# #### Limitations of the Metropolis Algorithm

# The limitations of the Metropolis-Hastings algorithm are mainly computational. With randomly generated proposals,
# it usually takes a large number of iterations to enter areas of higher (more likely) posterior densities. Even
# efficient Metropolis-Hastings algorithms sometimes accept less than 25% of the proposals (Roberts, Gelman & Gilks, 1997).
# In lower-dimensional situations, the increased computational power can compensate for the lower efficiency to some extent.
# But in higher-dimensional and more complex modeling situations, bigger and faster computers alone are rarely
# enough to overcome the challenge.

# #### Metropolis -- Implementation

# In our toy example we will assume that $J_t (\theta^* \mid \theta^{t-1})$ is symmetric, thus
# $J_t(\theta^* \mid \theta^{t-1}) = J_t (\theta^{t-1} \mid \theta^*)$, so I'll just implement
# the Metropolis algorithm (not the Metropolis-Hastings algorithm).

# Below I created a Metropolis sampler for our toy example. At the end it prints the acceptance rate of
# the proposals. Here I am using the same proposal distribution for both $X$ and $Y$: a uniform distribution
# parameterized with a `width` parameter:

# $$
# X \sim \text{Uniform} \left( X - \frac{\text{width}}{2}, X + \frac{\text{width}}{2} \right) \\
# Y \sim \text{Uniform} \left( Y - \frac{\text{width}}{2}, Y + \frac{\text{width}}{2} \right)
# $$

# I will use the already known `Distributions.jl` `MvNormal` from the plots above along with the `logpdf()`
# function to calculate the PDF of the proposed and current $\theta$s. It is easier to work with
# probability logs than with the absolute values[^numerical]. Mathematically we will compute:

# $$
# \begin{aligned}
# r &= \frac{
# \operatorname{PDF}\left(
# \text{Multivariate Normal} \left(
# \begin{bmatrix}
# x_{\text{proposed}} \\
# y_{\text{proposed}}
# \end{bmatrix}
# \right)
# \Bigg|
# \text{Multivariate Normal} \left(
# \begin{bmatrix}
# \mu_X \\
# \mu_Y
# \end{bmatrix}, \mathbf{\Sigma}
# \right)
# \right)}
# {
# \operatorname{PDF}\left(
# \text{Multivariate Normal} \left(
# \begin{bmatrix}
# x_{\text{current}} \\
# y_{\text{current}}
# \end{bmatrix}
# \right)
# \Bigg|
# \text{Multivariate Normal} \left(
# \begin{bmatrix}
# \mu_X \\
# \mu_Y
# \end{bmatrix}, \mathbf{\Sigma}
# \right)
# \right)}\\
# &=\frac{\operatorname{PDF}_{\text{proposed}}}{\operatorname{PDF}_{\text{current}}}\\
# &= \exp\Big(
# \log\left(\operatorname{PDF}_{\text{proposed}}\right)
# -
# \log\left(\operatorname{PDF}_{\text{current}}\right)
# \Big)
# \end{aligned}
# $$

# Here is a simple implementation in Julia:

function metropolis(S::Int64, width::Float64, ρ::Float64;
                    μ_x::Float64=0.0, μ_y::Float64=0.0,
                    σ_x::Float64=1.0, σ_y::Float64=1.0,
                    start_x=-2.5, start_y=2.5,
                    seed=123)
    rgn = MersenneTwister(seed)
    binormal = MvNormal([μ_x; μ_y], [σ_x ρ; ρ σ_y])
    draws = Matrix{Float64}(undef, S, 2)
    accepted = 0::Int64;
    x = start_x; y = start_y
    @inbounds draws[1, :] = [x y]
    for s in 2:S
        x_ = rand(rgn, Uniform(x - width, x + width))
        y_ = rand(rgn, Uniform(y - width, y + width))
        r = exp(logpdf(binormal, [x_, y_]) - logpdf(binormal, [x, y]))

        if r > rand(rgn, Uniform())
            x = x_
            y = y_
            accepted += 1
        end
        @inbounds draws[s, :] = [x y]
    end
    println("Acceptance rate is: $(accepted / S)")
    return draws
end

# Now let's run our `metropolis()` algorithm for the bivariate normal case with
# `S = 10_000`, `width = 2.75` and `ρ = 0.8`:

const S = 10_000
const width = 2.75
const ρ = 0.8

X_met = metropolis(S, width, ρ);

# Take a quick peek into `X_met`, we'll see it's a matrix of $X$ and $Y$ values as columns and the time $t$ as rows:

X_met[1:10, :]

# Also note that the acceptance of the proposals was 21%, the expected for Metropolis algorithms (around 20-25%)
# (Roberts et. al, 1997).

# We can construct `Chains` object using `MCMCChains.jl`[^mcmcchains] by passing a matrix along with the parameters names as
# symbols inside the `Chains()` constructor:

using MCMCChains

chain_met = Chains(X_met, [:X, :Y]);

# Then we can get summary statistics regarding our Markov chain derived from the Metropolis algorithm:

summarystats(chain_met)

# Both of `X` and `Y` have mean close to 0 and standard deviation close to 1 (which
# are the theoretical values).
# Take notice of the `ess` (effective sample size - ESS) is approximate 1,000.
# So let's calculate the efficiency of our Metropolis algorithm by dividing
# the ESS by the number of sampling iterations that we've performed:

# $$ \text{efficiency} = \frac{\text{ESS}}{\text{iterations}} \label{ESS} $$

mean(summarystats(chain_met)[:, :ess]) / S

# Our Metropolis algorithm has around 10.2% efficiency. Which, in my honest opinion, *sucks*...(😂)

# ##### Metropolis -- Visual Intuition

# I believe that a good visual intuition, even if you have not understood any mathematical formula, is the key for you to start a
# fruitful learning journey. So I made some animations!

# The animation in figure below shows the first 100 simulations of the Metropolis algorithm used to generate `X_met`.
# Note that in several iterations the proposal is rejected and the algorithm samples the parameters $\theta_1$ and $\theta_2$
# from the previous state (which becomes the current one, since the proposal is refused). The blue-filled ellipsis represents
# the 90% HPD of our toy example's bivariate normal distribution.

# Note: `HPD` stands for *Highest Probability Density* (which in our case the posterior's 90% probability range).

plt = covellipse(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.3,
    c=:steelblue,
    label="90% HPD",
    xlabel=L"\theta_1", ylabel=L"\theta_2")

met_anim = @animate for i in 1:100
    scatter!(plt, (X_met[i, 1], X_met[i, 2]),
             label=false, mc=:red, ma=0.5)
    plot!(X_met[i:i + 1, 1], X_met[i:i + 1, 2], seriestype=:path,
          lc=:green, la=0.5, label=false)
end
gif(met_anim, joinpath(@OUTPUT, "met_anim.gif"), fps=5); # hide

# \fig{met_anim}
# \center{*Animation of the First 100 Samples Generated from the Metropolis Algorithm*} \\

# Now let's take a look how the first 1,000 simulations were, excluding 1,000 initial iterations as warm-up.

const warmup = 1_000

scatter((X_met[warmup:warmup + 1_000, 1], X_met[warmup:warmup + 1_000, 2]),
         label=false, mc=:red, ma=0.3,
         xlims=(-3, 3), ylims=(-3, 3),
         xlabel=L"\theta_1", ylabel=L"\theta_2")

covellipse!(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.5,
    c=:steelblue,
    label="90% HPD")


savefig(joinpath(@OUTPUT, "met_first1000.svg")); # hide

# \fig{met_first1000}
# \center{*First 1,000 Samples Generated from the Metropolis Algorithm after warm-up*} \\

# And, finally, lets take a look in the all 9,000 samples generated after the warm-up of 1,000 iterations.

scatter((X_met[warmup:end, 1], X_met[warmup:end, 2]),
         label=false, mc=:red, ma=0.3,
         xlims=(-3, 3), ylims=(-3, 3),
         xlabel=L"\theta_1", ylabel=L"\theta_2")

covellipse!(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.5,
    c=:steelblue,
    label="90% HPD")
savefig(joinpath(@OUTPUT, "met_all.svg")); # hide

# \fig{met_all}
# \center{*All 9,000 Samples Generated from the Metropolis Algorithm after warm-up*} \\

# ### Gibbs

# To circumvent the problem of low acceptance rate of the Metropolis (and Metropolis-Hastings) algorithm,
# the Gibbs algorithm was developed, which does not have an acceptance/rejection rule for new proposals
# to change the state of the Markov chain. **All proposals are accepted**.

# Gibbs' algorithm had an original idea conceived by physicist Josiah Willard Gibbs (figure below),
# in reference to an analogy between a sampling algorithm and statistical physics (a branch of physics which
# is based on statistical mechanics). The algorithm was described by brothers Stuart and Donald Geman
# in 1984 (Geman & Geman, 1984), about eight decades after Gibbs's death.

# ![Josiah Gibbs](/pages/images/josiah_gibbs.jpg)
#
# \center{*Josiah Gibbs*} \\

# The Gibbs algorithm is very useful in multidimensional sample spaces (in which there are more than 2 parameters to be
# sampled for the posterior probability). It is also known as **alternating conditional sampling**, since we always sample
# a parameter **conditioned** to the probability of the other parameters.

# The Gibbs algorithm can be seen as a **special case** of the Metropolis-Hastings algorithm because all proposals are
# accepted (Gelman, 1992).

# #### Gibbs Algorithm

# The essence of Gibbs' algorithm is an iterative sampling of parameters conditioned to other parameters
# $P(\theta_1 \mid \theta_2, \dots \theta_n)$.

# Gibbs's algorithm can be described in the following way[^gibbs] ($\theta$ is the parameter, or set of
# parameters, of interest and $y$ is the data):

# 1. Define $P(\theta_1), P(\theta_2), \dots, P(\theta_n)$: the prior probability of each of the $\theta_n$ parameters.

# 2. Sample a starting point $\theta^0_1, \theta^0_2, \dots, \theta^0_n$. We usually sample from a normal distribution or from a distribution specified as the prior distribution of $\theta_n$.

# 3. For $t = 1, 2, \dots$:

#     $\begin{aligned}
#     \theta^t_1 &\sim p(\theta_1 \mid \theta^0_2, \dots, \theta^0_n) \\
#     \theta^t_2 &\sim p(\theta_2 \mid \theta^{t-1}_1, \dots, \theta^0_n) \\
#     &\vdots \\
#     \theta^t_n &\sim p(\theta_n \mid \theta^{t-1}_1, \dots, \theta^{t-1}_{n-1})
#     \end{aligned}$

# #### Limitations of the Gibbs Algorithm

# The main limitation of the Gibbs algorithm is with regard to alternative conditional sampling.

# If we compare with the Metropolis algorithm (and consequently Metropolis-Hastings) we have random proposals
# from a proposal distribution in which we sample each parameter unconditionally to other parameters. In order
# for the proposals to take us to the posterior probability's correct locations to sample, we have an
# acceptance/rejection rule for these proposals, otherwise the samples of the Metropolis algorithm would not
# approach the target distribution of interest. The state changes of the Markov chain are then carried out
# multidimensionally [^gibbs2]. As you saw in the Metropolis' figures, in a 2-D space (as is our example's bivariate
# normal distribution), when there is a change of state in the Markov chain, the new proposal location considers
# both $\theta_1$ and $\theta_2$, causing **diagonal** movement in space 2-D sample. In other words,
# the proposal is done regarding all dimensions of the parameter space.

# In the case of the Gibbs algorithm, in our toy example, this movement occurs only in a single parameter,
# *i.e* single dimension, as we sample sequentially and conditionally to other parameters. This causes **horizontal**
# movements (in the case of $\theta_1$) and **vertical movements** (in the case of $\theta_2$), but never
# diagonal movements like the ones we saw in the Metropolis algorithm.

# #### Gibbs -- Implementation

# Here are some new things compared to the Metropolis algorithm implementation. First to conditionally
# sample the parameters $P(\theta_1 \mid \theta_2)$ and $P(\theta_2 \mid \theta_1)$, we need to create
# two new variables `β` and `λ`. These variables represent the correlation between $X$ and $Y$ ($\theta_1$
# and $\theta_2$ respectively) scaled by the ratio of the variance of $X$ and $Y$.
# And then we use these variables in the sampling of $\theta_1$ and $\theta_2$:

# $$
# \begin{aligned}
# \beta &= \rho \cdot \frac{\sigma_Y}{\sigma_X} = \rho \\
# \lambda &= \rho \cdot \frac{\sigma_X}{\sigma_Y} = \rho \\
# \sigma_{YX} &= 1 - \rho^2\\
# \sigma_{XY} &= 1 - \rho^2\\
# \theta_1 &\sim \text{Normal} \bigg( \mu_X + \lambda \cdot (y^* - \mu_Y), \sigma_{XY} \bigg) \\
# \theta_2 &\sim \text{Normal} \bigg( \mu_y + \beta \cdot (x^* - \mu_X), \sigma_{YX} \bigg)
# \end{aligned}
# $$

function gibbs(S::Int64, ρ::Float64;
               μ_x::Float64=0.0, μ_y::Float64=0.0,
               σ_x::Float64=1.0, σ_y::Float64=1.0,
               start_x=-2.5, start_y=2.5,
               seed=123)
    rgn = MersenneTwister(seed)
    binormal = MvNormal([μ_x; μ_y], [σ_x ρ; ρ σ_y])
    draws = Matrix{Float64}(undef, S, 2)
    x = start_x; y = start_y
    β = ρ * σ_y / σ_x
    λ = ρ * σ_x / σ_y
    sqrt1mrho2 = sqrt(1 - ρ^2)
    σ_YX = σ_y * sqrt1mrho2
    σ_XY = σ_x * sqrt1mrho2
    @inbounds draws[1, :] = [x y]
    for s in 2:S
        if s % 2 == 0
            y = rand(rgn, Normal(μ_y + β * (x - μ_x), σ_YX))
        else
            x = rand(rgn, Normal(μ_x + λ * (y - μ_y), σ_XY))
        end
        @inbounds draws[s, :] = [x y]
    end
    return draws
end

# Generally a Gibbs sampler is not implemented in this way. Here I coded the Gibbs algorithm so that it samples a parameter for each iteration.
# To be more computationally efficient we would sample all parameters are on each iteration. I did it on purpose because I want
# to show in the animations the real trajectory of the Gibbs sampler in the sample space (vertical and horizontal, not diagonal).
# So to remedy this I will provide `gibbs()` double the ammount of `S` (20,000 in total). Also take notice that we are now proposing
# new parameters' values conditioned on other parameters, so there is not an acceptance/rejection rule here.

X_gibbs = gibbs(S * 2, ρ);

# As before lets' take a quick peek into `X_gibbs`, we'll see it's a matrix of $X$ and $Y$ values as columns and the time $t$ as rows:

X_gibbs[1:10, :]

# Again, we construct a `Chains` object by passing a matrix along with the parameters names as
# symbols inside the `Chains()` constructor:

chain_gibbs = Chains(X_gibbs, [:X, :Y]);

# Then we can get summary statistics regarding our Markov chain derived from the Gibbs algorithm:

summarystats(chain_gibbs)

# Both of `X` and `Y` have mean close to 0 and standard deviation close to 1 (which
# are the theoretical values).
# Take notice of the `ess` (effective sample size - ESS) that is around 2,100.
# Since we used `S * 2` as the number of samples, in order for we to compare with Metropolis,
# we would need to divide the ESS by 2. So our ESS is between 1,000, which is similar
# to Metropolis' ESS.
# Now let's calculate the efficiency of our Gibbs algorithm by dividing
# the ESS by the number of sampling iterations that we've performed also
# accounting for the `S * 2`:

(mean(summarystats(chain_gibbs)[:, :ess]) / 2) / S

# Our Gibbs algorithm has around 10.6% efficiency. Which, in my honest opinion, despite the
# small improvement still *sucks*...(😂)

# ##### Gibbs -- Visual Intuition

# Oh yes, we have animations for Gibbs also!

# The animation in figure below shows the first 100 simulations of the Gibbs algorithm used to generate `X_gibbs`.
# Note that all proposals are accepted now, so the at each iteration we sample new parameters values.
# The blue-filled ellipsis represents the 90% HPD of our toy example's bivariate normal distribution.

plt = covellipse(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.3,
    c=:steelblue,
    label="90% HPD",
    xlabel=L"\theta_1", ylabel=L"\theta_2")

gibbs_anim = @animate for i in 1:200
    scatter!(plt, (X_gibbs[i, 1], X_gibbs[i, 2]),
             label=false, mc=:red, ma=0.5)
    plot!(X_gibbs[i:i + 1, 1], X_gibbs[i:i + 1, 2], seriestype=:path,
          lc=:green, la=0.5, label=false)
end
gif(gibbs_anim, joinpath(@OUTPUT, "gibbs_anim.gif"), fps=5); # hide

# \fig{gibbs_anim}
# \center{*Animation of the First 100 Samples Generated from the Gibbs Algorithm*} \\

# Now let's take a look how the first 1,000 simulations were, excluding 1,000 initial iterations as warm-up.

scatter((X_gibbs[2 * warmup:2 * warmup + 1_000, 1], X_gibbs[2 * warmup:2 * warmup + 1_000, 2]),
         label=false, mc=:red, ma=0.3,
         xlims=(-3, 3), ylims=(-3, 3),
         xlabel=L"\theta_1", ylabel=L"\theta_2")

covellipse!(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.5,
    c=:steelblue,
    label="90% HPD")


savefig(joinpath(@OUTPUT, "gibbs_first1000.svg")); # hide

# \fig{gibbs_first1000}
# \center{*First 1,000 Samples Generated from the Gibbs Algorithm after warm-up*} \\

# And, finally, lets take a look in the all 9,000 samples generated after the warm-up of 1,000 iterations.

scatter((X_gibbs[2 * warmup:end, 1], X_gibbs[2 * warmup:end, 2]),
         label=false, mc=:red, ma=0.3,
         xlims=(-3, 3), ylims=(-3, 3),
         xlabel=L"\theta_1", ylabel=L"\theta_2")

covellipse!(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.5,
    c=:steelblue,
    label="90% HPD")
savefig(joinpath(@OUTPUT, "gibbs_all.svg")); # hide

# \fig{gibbs_all}
# \center{*All 9,000 Samples Generated from the Gibbs Algorithm after warm-up*} \\

# ### What happens when we run Markov chains in parallel?

# Since the Markov chains are **independent**, we can run them in **parallel**. The key to this is
# **defining different starting points for each Markov chain** (if you use a sample of a previous distribution
# of parameters as a starting point this is not a problem). We will use the same toy example of a bivariate normal
# distribution $X$ and $Y$ that we used in the previous examples, but now with **4 Markov chains in parallel
# with different starting points**[^markovparallel].

# First, let's defined 4 different pairs of starting points using a nice Cartesian product
# from Julia's `Base.Iterators`:

const starts = Iterators.product((-2.5, 2.5), (2.5, -2.5)) |> collect

# Also, I will restrict this simulation to 100 samples:

const S_parallel = 100;

# Additionally, note that we are using different `seed`s:

X_met_1 = metropolis(S_parallel, width, ρ, seed=124, start_x=first(starts[1]), start_y=last(starts[1]));
X_met_2 = metropolis(S_parallel, width, ρ, seed=125, start_x=first(starts[2]), start_y=last(starts[2]));
X_met_3 = metropolis(S_parallel, width, ρ, seed=126, start_x=first(starts[3]), start_y=last(starts[3]));
X_met_4 = metropolis(S_parallel, width, ρ, seed=127, start_x=first(starts[4]), start_y=last(starts[4]));

# There have been some significant changes in the approval rate for Metropolis proposals. All were around 13%-24%,
# this is due to the low number of samples (only 100 for each Markov chain), if the samples were larger we would see
# these values converge to close to 20% according to the previous example of 10,000 samples with a single stream
# (Roberts et. al, 1997).

# Now let's take a look on how those 4 Metropolis Markov chains sample the parameter space starting from different positions.
# Each chain will have its own marker and path color, so that we can see their different behavior:

plt = covellipse(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.3,
    c=:grey,
    label="90% HPD",
    xlabel=L"\theta_1", ylabel=L"\theta_2")

const logocolors = Colors.JULIA_LOGO_COLORS;

parallel_met = Animation()
for i in 1:99
    scatter!(plt, (X_met_1[i, 1], X_met_1[i, 2]),
             label=false, mc=logocolors.blue, ma=0.5)
    plot!(X_met_1[i:i + 1, 1], X_met_1[i:i + 1, 2], seriestype=:path,
          lc=logocolors.blue, la=0.5, label=false)
    scatter!(plt, (X_met_2[i, 1], X_met_2[i, 2]),
             label=false, mc=logocolors.red, ma=0.5)
    plot!(X_met_2[i:i + 1, 1], X_met_2[i:i + 1, 2], seriestype=:path,
          lc=logocolors.red, la=0.5, label=false)
    scatter!(plt, (X_met_3[i, 1], X_met_3[i, 2]),
             label=false, mc=logocolors.green, ma=0.5)
    plot!(X_met_3[i:i + 1, 1], X_met_3[i:i + 1, 2], seriestype=:path,
          lc=logocolors.green, la=0.5, label=false)
    scatter!(plt, (X_met_4[i, 1], X_met_4[i, 2]),
             label=false, mc=logocolors.purple, ma=0.5)
    plot!(X_met_4[i:i + 1, 1], X_met_4[i:i + 1, 2], seriestype=:path,
          lc=logocolors.purple, la=0.5, label=false)
    frame(parallel_met)
end
gif(parallel_met, joinpath(@OUTPUT, "parallel_met.gif"), fps=5); # hide

# \fig{parallel_met}
# \center{*Animation of 4 Parallel Metropolis Markov Chains*} \\

# Now we'll do the the same for Gibbs, taking care to provide also different `seed`s and starting points:

X_gibbs_1 = gibbs(S_parallel * 2, ρ, seed=124, start_x=first(starts[1]), start_y=last(starts[1]));
X_gibbs_2 = gibbs(S_parallel * 2, ρ, seed=125, start_x=first(starts[2]), start_y=last(starts[2]));
X_gibbs_3 = gibbs(S_parallel * 2, ρ, seed=126, start_x=first(starts[3]), start_y=last(starts[3]));
X_gibbs_4 = gibbs(S_parallel * 2, ρ, seed=127, start_x=first(starts[4]), start_y=last(starts[4]));

# Now let's take a look on how those 4 Gibbs Markov chains sample the parameter space starting from different positions.
# Each chain will have its own marker and path color, so that we can see their different behavior:

plt = covellipse(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.3,
    c=:grey,
    label="90% HPD",
    xlabel=L"\theta_1", ylabel=L"\theta_2")

parallel_gibbs = Animation()
for i in 1:199
    scatter!(plt, (X_gibbs_1[i, 1], X_gibbs_1[i, 2]),
             label=false, mc=logocolors.blue, ma=0.5)
    plot!(X_gibbs_1[i:i + 1, 1], X_gibbs_1[i:i + 1, 2], seriestype=:path,
          lc=logocolors.blue, la=0.5, label=false)
    scatter!(plt, (X_gibbs_2[i, 1], X_gibbs_2[i, 2]),
             label=false, mc=logocolors.red, ma=0.5)
    plot!(X_gibbs_2[i:i + 1, 1], X_gibbs_2[i:i + 1, 2], seriestype=:path,
          lc=logocolors.red, la=0.5, label=false)
    scatter!(plt, (X_gibbs_3[i, 1], X_gibbs_3[i, 2]),
             label=false, mc=logocolors.green, ma=0.5)
    plot!(X_gibbs_3[i:i + 1, 1], X_gibbs_3[i:i + 1, 2], seriestype=:path,
          lc=logocolors.green, la=0.5, label=false)
    scatter!(plt, (X_gibbs_4[i, 1], X_gibbs_4[i, 2]),
             label=false, mc=logocolors.purple, ma=0.5)
    plot!(X_gibbs_4[i:i + 1, 1], X_gibbs_4[i:i + 1, 2], seriestype=:path,
          lc=logocolors.purple, la=0.5, label=false)
    frame(parallel_gibbs)
end
gif(parallel_gibbs, joinpath(@OUTPUT, "parallel_gibbs.gif"), fps=5); # hide

# \fig{parallel_gibbs}
# \center{*Animation of 4 Parallel Gibbs Markov Chains*} \\

# ## Hamiltonian Monte Carlo -- HMC

# The problems of low acceptance rates of proposals for Metropolis techniques and the low performance of the Gibbs algorithm
# in multidimensional problems (in which the posterior's topology is quite complex) led to the emergence of a new MCMC technique
# using Hamiltonian dynamics (in honor of the Irish physicist [ William Rowan Hamilton](https://en.wikipedia.org/wiki/William_Rowan_Hamilton)
# (1805-1865) (figure below). The name for this technique is *Hamiltonian Monte Carlo* -- HMC.

# ![William Rowan Hamilton](/pages/images/hamilton.png)
#
# \center{*William Rowan Hamilton*} \\

# The HMC is an adaptation of the Metropolis technique and employs a guided scheme for generating new proposals:
# this improves the proposal's acceptance rate and, consequently, efficiency. More specifically, the HMC uses
# the posterior log gradient to direct the Markov chain to regions of higher posterior density, where most samples
# are collected. As a result, a Markov chain with the well-adjusted HMC algorithm will accept proposals at a much higher
# rate than the traditional Metropolis algorithm (Roberts et. al, 1997).

# HMC was first described in the physics literature (Duane, Kennedy, Pendleton & Roweth, 1987) (which they called the
# *"Hybrid" Monte Carlo* -- HMC). Soon after, HMC was applied to statistical problems by Neal (1994) (which he called
# *Hamiltonean Monte Carlo* -- HMC). For an in-depth discussion regarding HMC (which is not the focus of this tutorial),
# I recommend Neal (2011) and Betancourt (2017).

# HMC uses Hamiltonian dynamics applied to particles exploring the topology of a posterior density. In some simulations
# Metropolis has an acceptance rate of approximately 23%, while HMC 65% (Gelman et al., 2013b). In addition to better
# exploring the posterior's topology and tolerating complex topologies, HMC is much more efficient than Metropolis and
# does not suffer from the Gibbs' parameter correlation problem.

# For each component $\theta_j$, the HMC adds a momentum variable $\phi_j$. The subsequent density $P(\theta \mid y)$
# is increased by an independent distribution $P(\phi)$ of the momentum, thus defining a joint distribution:

# $$ P(\theta, \phi \mid y) = P(\phi) \cdot P(\theta \mid y) \label{hmcjoint} $$

# HMC uses a proposal distribution that changes depending on the current state in the Markov chain. The HMC discovers
# the direction in which the posterior distribution increases, called *gradient*, and distorts the distribution of proposals
# towards the *gradient*. In the Metropolis algorithm, the distribution of the proposals would be a (usually) Normal distribution
# centered on the current position, so that jumps above or below the current position would have the same probability of being
# proposed. But the HMC generates proposals quite differently.

# You can imagine that for high-dimensional posterior densities that have *narrow diagonal valleys* and even *curved valleys*,
# the HMC dynamics will find proposed positions that are much more **promising** than a naive symmetric proposal distribution,
# and more promising than the Gibbs sampling, which can get stuck in *diagonal walls*.

# The probability of the Markov chain changing state in the HMC algorithm is defined as:

# $$
# P_{\text{change}} = \min\left({\frac{P(\theta_{\text{proposed}}) \cdot
# P(\phi_{\text{proposed}})}{P(\theta_{\text{current}})\cdot P(\phi_{\text{current}})}}, 1\right) \label{hmcproposal}
# $$

# where $\phi$ is the momentum.

# ### Momentum Distribution -- $P(\phi)$

# We normally give $\phi$ a normal multivariate distribution with a mean of 0 and a covariance of $\mathbf{M}$, a "mass matrix".
# To keep things a little bit simpler, we use a diagonal mass matrix $\mathbf{M}$. This makes the components of $\phi$ independent
# with $\phi_j \sim \text{Normal}(0, M_{jj})$

# ### HMC Algorithm

# The HMC algorithm is very similar to the Metropolis algorithm but with the inclusion of the momentum $\phi$ as a way of
# quantifying the gradient of the posterior distribution:

# 1. Sample $\phi$ from a $\text{Normal}(0, \mathbf{M})$

# 2. Simultaneously sample $\theta$ and $\phi$ with $L$ *leapfrog steps* each scaled by a $\epsilon$ factor. In a *leapfrog step*, both $\theta$ and $\phi$ are changed, in relation to each other. Repeat the following steps $L$ times:

# 2.1. Use the gradient of log posterior of $\theta$ to produce a *half-step* of $\phi$:

#         $$\phi \leftarrow \phi + \frac{1}{2} \epsilon \frac{d \log p(\theta \mid y)}{d \theta}$$

# 2.2 Use the momentum vector $\phi$ to update the parameter vector $\theta$:

#         $$ \theta \leftarrow \theta + \epsilon \mathbf{M}^{-1} \phi $$

# 2.3. Use again the gradient of log posterior of $\theta$ to another *half-step* of $\phi$:

#         $$ \phi \leftarrow \phi + \frac{1}{2} \epsilon \frac{d \log p(\theta \mid y)}{d \theta} $$

# 3. Assign $\theta^{t-1}$ and $\phi^{t-1}$ as the values of the parameter vector and the momentum vector, respectively, at the beginning of the *leapfrog* process (step 2) and $\theta^*$ and $\phi^*$ as the values after $L$ steps. As an acceptance/rejection rule calculate:

#     $$ r = \frac{p(\theta^* \mid y) p(\phi^*)}{p(\theta^{t-1} \mid y) p(\phi^{-1})} $$

# 4. Assign:

#      $$\theta^t =
#      \begin{cases}
#      \theta^* & \text{with probability } \min (r, 1) \\
#      \theta^{t-1} & \text{otherwise}
#      \end{cases}$$


# ### HMC -- Implementation

# Alright let's code the HMC algorithm for our toy example's bivariate normal distribution:

using ForwardDiff:gradient
function hmc(S::Int64, width::Float64, ρ::Float64;
             L=40, ϵ=0.001,
             μ_x::Float64=0.0, μ_y::Float64=0.0,
             σ_x::Float64=1.0, σ_y::Float64=1.0,
             start_x=-2.5, start_y=2.5,
             seed=123)
    rgn = MersenneTwister(seed)
    binormal = MvNormal([μ_x; μ_y], [σ_x ρ; ρ σ_y])
    draws = Matrix{Float64}(undef, S, 2)
    accepted = 0::Int64;
    x = start_x; y = start_y
    @inbounds draws[1, :] = [x y]
    M = [1. 0.; 0. 1.]
    ϕ_d = MvNormal([0.0, 0.0], M)
    for s in 2:S
        x_ = rand(rgn, Uniform(x - width, x + width))
        y_ = rand(rgn, Uniform(y - width, y + width))
        ϕ = rand(rgn, ϕ_d)
        kinetic = sum(ϕ.^2) / 2
        log_p = logpdf(binormal, [x, y]) - kinetic
        ϕ += 0.5 * ϵ * gradient(x -> logpdf(binormal, x), [x_, y_])
        for l in 1:L
            x_, y_ = [x_, y_] + (ϵ * M * ϕ)
            ϕ += + 0.5 * ϵ * gradient(x -> logpdf(binormal, x), [x_, y_])
        end
        ϕ = -ϕ # make the proposal symmetric
        kinetic = sum(ϕ.^2) / 2
        log_p_ = logpdf(binormal, [x_, y_]) - kinetic
        r = exp(log_p_ - log_p)

        if r > rand(rgn, Uniform())
            x = x_
            y = y_
            accepted += 1
        end
        @inbounds draws[s, :] = [x y]
    end
    println("Acceptance rate is: $(accepted / S)")
    return draws
end

# In the `hmc()` function above I am using the `gradient()` function from
# [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) (Revels, Lubin & Papamarkou, 2016)
# which is Julia's package for forward mode auto differentiation (autodiff).
# The `gradient()` function accepts a function as input and an array $\mathbf{X}$. It literally evaluates the function $f$
# at $\mathbf{X}$ and returns the gradient $\nabla f(\mathbf{X})$.
# This is one the advantages of Julia: I don't need to implement an autodiff for `logpdf()`s of any distribution, it will
# be done automatically for any one from `Distributions.jl`. You can also try reverse mode autodiff with
# [`ReverseDiff.jl`](https://github.com/JuliaDiff/ReverseDiff.jl) if you want to; and it will also very easy to get a gradient.
# Now, we've got carried away with Julia's amazing autodiff potential... Let me show you an example of a gradient of a log PDF
# evaluated at some value. I will use our `mvnormal` bivariate normal distribution as an example and evaluate its gradient
# at $x = 1$ and $y = -1$:

gradient(x -> logpdf(mvnormal, x), [1, -1])

# So the gradient tells me that the partial derivative of $x = 1$ with respect to our `mvnormal` distribution is `-5`
# and the partial derivative of $y = -1$ with respect to our `mvnormal` distribution is `5`.

# Now let's run our HMC Markov chain.
# We are going to use $L = 40$ and (don't ask me how I found out) $\epsilon = 0.0856$:

X_hmc = hmc(S, width, ρ, ϵ=0.0856, L=40);

# Our acceptance rate is 20.79%.
# As before lets' take a quick peek into `X_hmc`, we'll see it's a matrix of $X$ and $Y$ values as columns and the time $t$ as rows:

X_hmc[1:10, :]

# Again, we construct a `Chains` object by passing a matrix along with the parameters names as
# symbols inside the `Chains()` constructor:

chain_hmc = Chains(X_hmc, [:X, :Y]);

# Then we can get summary statistics regarding our Markov chain derived from the HMC algorithm:

summarystats(chain_hmc)

# Both of `X` and `Y` have mean close to 0 and standard deviation close to 1 (which
# are the theoretical values).
# Take notice of the `ess` (effective sample size - ESS) that is around 1,600.
# Now let's calculate the efficiency of our HMC algorithm by dividing
# the ESS by the number of sampling iterations:

mean(summarystats(chain_hmc)[:, :ess]) / S

# We see that a simple naïve (and not well-calibrated[^calibrated]) HMC has 70% more efficiency from both Gibbs and Metropolis.
# ≈ 10% versus ≈ 17%. Great! 😀

# #### HMC -- Visual Intuition

# This wouldn't be complete without animations for HMC!

# The animation in figure below shows the first 100 simulations of the HMC algorithm used to generate `X_hmc`.
# Note that we have a gradient-guided proposal at each iteration, so the animation would resemble more like
# a very lucky random-walk Metropolis [^luckymetropolis].
# The blue-filled ellipsis represents the 90% HPD of our toy example's bivariate normal distribution.

plt = covellipse(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.3,
    c=:steelblue,
    label="90% HPD",
    xlabel=L"\theta_1", ylabel=L"\theta_2")

hmc_anim = @animate for i in 1:100
    scatter!(plt, (X_hmc[i, 1], X_hmc[i, 2]),
             label=false, mc=:red, ma=0.5)
    plot!(X_hmc[i:i + 1, 1], X_hmc[i:i + 1, 2], seriestype=:path,
          lc=:green, la=0.5, label=false)
end
gif(hmc_anim, joinpath(@OUTPUT, "hmc_anim.gif"), fps=5); # hide

# \fig{hmc_anim}
# \center{*Animation of the First 100 Samples Generated from the HMC Algorithm*} \\

# As usual, let's take a look how the first 1,000 simulations were, excluding 1,000 initial iterations as warm-up.

scatter((X_hmc[warmup:warmup + 1_000, 1], X_hmc[warmup:warmup + 1_000, 2]),
         label=false, mc=:red, ma=0.3,
         xlims=(-3, 3), ylims=(-3, 3),
         xlabel=L"\theta_1", ylabel=L"\theta_2")

covellipse!(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.5,
    c=:steelblue,
    label="90% HPD")


savefig(joinpath(@OUTPUT, "hmc_first1000.svg")); # hide

# \fig{hmc_first1000}
# \center{*First 1,000 Samples Generated from the HMC Algorithm after warm-up*} \\


# And, finally, lets take a look in the all 9,000 samples generated after the warm-up of 1,000 iterations.

scatter((X_hmc[warmup:end, 1], X_hmc[warmup:end, 2]),
         label=false, mc=:red, ma=0.3,
         xlims=(-3, 3), ylims=(-3, 3),
         xlabel=L"\theta_1", ylabel=L"\theta_2")

covellipse!(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-3, 3), ylims=(-3, 3),
    alpha=0.5,
    c=:steelblue,
    label="90% HPD")
savefig(joinpath(@OUTPUT, "hmc_all.svg")); # hide

# \fig{hmc_all}
# \center{*All 9,000 Samples Generated from the HMC Algorithm after warm-up*} \\

# ### HMC -- Complex Topologies

# There are cases where HMC will be much better than Metropolis or Gibbs. In particular, these cases focus on complicated
# and difficult-to-explore posterior topologies. In these contexts, an algorithm that can guide the proposals to regions
# of higher density (such as the case of the HMC) is able to explore much more efficient (less iterations for convergence)
# and effective (higher rate of acceptance of the proposals).

# See figure below for an example of a bimodal posterior density with also the marginal histogram of $X$ and $Y$:

# $$
# X = \text{Normal} \left(
# \begin{bmatrix}
# 10 \\
# 2
# \end{bmatrix},
# \begin{bmatrix}
# 1 & 0 \\
# 0 & 1
# \end{bmatrix}
# \right), \quad
# Y = \text{Normal} \left(
# \begin{bmatrix}
# 0 \\
# 0
# \end{bmatrix},
# \begin{bmatrix}
# 8.4 & 2.0 \\
# 2.0 & 1.7
# \end{bmatrix}
# \right)
# $$

d1 = MvNormal([10, 2], [1 0; 0 1])
d2 = MvNormal([0, 0], [8.4 2.0; 2.0 1.7])

d = MixtureModel([d1, d2])

data_mixture = rand(d, 1_000)'

marginalkde(data_mixture[:, 1], data_mixture[:, 2],
            clip=((-1.6, 3), (-3, 3)),
            xlabel=L"X", ylabel=L"Y")
savefig(joinpath(@OUTPUT, "bimodal.svg")); # hide

# \fig{bimodal}
# \center{*Multivariate Bimodal Normal*} \\

# And to finish an example of Neal's funnel Neal(2003) in the figure below. This is a very difficult posterior to be sampled
# even for HMC, as it varies in geometry in the dimensions $X$ and $Y$. This means that the HMC sampler has to change the
# *leapfrog steps* $L$ and the scaling factor $\epsilon$ every time, since at the top of the image (the top of the funnel)
# a large value of $L$ is needed along with a small $\epsilon$; and at the bottom (at the bottom of the funnel) the opposite:
# small $L$ and large $\epsilon$.

funnel_y = rand(Normal(0, 3), 10_000)
funnel_x = rand(Normal(), 10_000) .* exp.(funnel_y / 2)

scatter((funnel_x, funnel_y),
        label=false, mc=:steelblue, ma=0.3,
        xlabel=L"X", ylabel=L"Y",
        xlims=(-100, 100))
savefig(joinpath(@OUTPUT, "funnel.svg")); # hide

# \fig{funnel}
# \center{*Neal's Funnel*} \\

# ## "I understood nothing..."

# If you haven't understood anything by now, don't despair. Skip all the formulas and get the visual intuition of the algorithms.
# See the limitations of Metropolis and Gibbs and compare the animations and figures with those of the HMC. The superiority
# of efficiency (more samples with low autocorrelation - ESS) and effectiveness (more samples close to the most likely areas
# of the target distribution) is self-explanatory by the images.

# In addition, you will probably **never** have to code your HMC algorithm (Gibbs, Metropolis or any other MCMC) by hand.
# For that there are packages like Turing's [`AdvancedHMC.jl`](https://github.com/TuringLang/AdvancedHMC.jl)
# In addition, `AdvancedHMC` has a modified HMC with a technique called **N**o-**U**-**T**urn **S**ampling (NUTS)[^nuts]
# (Hoffman & Gelman, 2011) that selects automatically the values ​​of $\epsilon$ (scaling factor) and $L$ (*leapfrog steps*).
# The performance of the HMC is highly sensitive to these two "hyperparameters" (parameters that must be specified by the user).
# In particular, if $L$ is too small, the algorithm exhibits undesirable behavior of a random walk, while if $L$ is too large,
# the algorithm wastes computational efficiency. NUTS uses a recursive algorithm to build a set of likely candidate points that span
# a wide range of proposal distribution, automatically stopping when it starts to go back and retrace its steps
# (why it doesn't turn around - *No U-turn*), in addition NUTS also automatically calibrates simultaneously $L$ and $\epsilon$.

# ## MCMC Metrics

# All Markov chains have some convergence and diagnostics metrics that you should be aware of. Note that this is still an ongoing
# are of intense research and new metrics are constantly being proposed (*e.g.* Vehtari, Gelman., Simpson, Carpenter & Bürkner (2021))
# To show MCMC metrics let me recover our six-sided dice example from [4. **How to use Turing**](/pages/4_Turing/):

using Turing
setprogress!(false) # hide

@model dice_throw(y) = begin
    #Our prior belief about the probability of each result in a six-sided dice.
    #p is a vector of length 6 each with probability p that sums up to 1.
    p ~ Dirichlet(6, 1)

    #Each outcome of the six-sided dice has a probability p.
    for i in eachindex(y)
        y[i] ~ Categorical(p)
    end
end;

# Let's once again generate 1,000 throws of a six-sided dice:

data_dice = rand(DiscreteUniform(1, 6), 1_000);

# Like before we'll sample using `NUTS()` and 2,000 iterations. Note that you can use Metropolis with `MH()`, Gibbs with `Gibbs()`
# and HMC with `HMC()` if you want to. You can check all Turing's different MCMC samplers in
# [Turing's documentation](https://turing.ml/dev/docs/using-turing/guide).

model = dice_throw(data_dice)
chain = sample(model, NUTS(), 2_000);
summarystats(chain)

# We have the following columns that outpus some kind of MCMC summary statistics:

# * `mcse`: **M**onte **C**arlo **S**tandard **E**rror, the uncertainty about a statistic in the sample due to sampling error.
# * `ess`: **E**ffective **S**ample **S**ize, a rough approximation of the number of effective samples sampled by the MCMC estimated by the value of` rhat`.
# * `rhat`: a metric of convergence and stability of the Markov chain.

# The most important metric to take into account is the 'rhat` which is a metric that measures whether the Markov chains
# are stable and converged to a value during the total progress of the sampling procedure. It is basically the proportion
# of variation when comparing two halves of the chains after discarding the warmups. A value of 1 implies convergence
# and stability. By default, `rhat` must be less than 1.01 for the Bayesian estimation to be valid
# (Brooks & Gelman, 1998; Gelman & Rubin, 1992).

# Note that all of our model's parameters have achieve a nice `ess` and, more important, `rhat` in the desired range, a solid
# indicator that the Markov chain is stable and has converged to the estimated parameter values.

# ### What to do if your model doesn't converge?

# Depending on the model and data, it is possible that HMC (even with NUTS) will not achieve convergence.
# NUTS will not converge if it encounters divergences either due to a very pathological posterior density topology
# or if you supply improper parameters. To exemplify let me run a "bad" chain by passing the `NUTS()` constructor
# a target acceptance rate of `0.3` with only 500 sampling iterations (including warmup):

bad_chain = sample(model, NUTS(0.3), 500)
summarystats(bad_chain)

# Here we can see that the `ess` and `rhat` of the `bad_chain` are *really* bad!
# There will be several divergences that we can access in the column `numerical_error` of a `Chains` object. Here we have 0 divergences.

sum(bad_chain[:numerical_error])

# Also we can see the Markov chain acceptance rate in the column `acceptance_rate`. This is the `bad_chain` acceptance rate:

mean(bad_chain[:acceptance_rate])

# And now the "good" `chain`:

mean(chain[:acceptance_rate])

# What a difference huh? 80% versus 0.5%.

# ## Footnotes
# [^propto]: the symbol $\propto$ (`\propto`) should be read as "proportional to".
# [^warmup]: some references call this process *burnin*.
# [^metropolis]: if you want a better explanation of the Metropolis and Metropolis-Hastings algorithms I suggest to see Chib & Greenberg (1995).
# [^numerical]: Due to easier computational complexity and to avoid [numeric overflow](https://en.wikipedia.org/wiki/Integer_overflow) we generally use sum of logs instead of multiplications, specially when dealing with probabilities, *i.e.* $\mathbb{R} \in [0, 1]$.
# [^mcmcchains]: this is one of the packages of Turing's ecosystem. I recommend you to take a look into [4. **How to use Turing**](/pages/4_Turing/).
# [^gibbs]: if you want a better explanation of the Gibbs algorithm I suggest to see Casella & George (1992).
# [^gibbs2]: this will be clear in the animations and images.
# [^markovparallel]: note that there is some shenanigans here to take care. You would also want to have different seeds for the random number generator in each Markov chain. This is why `metropolis()` and `gibbs()` have a `seed` parameter.
# [^calibrated]: as detailed in the following sections, HMC is quite sensitive to the choice of $L$ and $\epsilon$ and I haven't tried to get the best possible combination of those.
# [^luckymetropolis]: or a not-drunk random-walk Metropolis 😂.
# [^nuts]: NUTS is an algorithm that uses the symplectic leapfrog integrator and builds a binary tree composed of leaf nodes that are simulations of Hamiltonian dynamics using $2^j$ *leapfrog steps* in forward or backward directions in time where $j$ is the integer representing the iterations of the construction of the binary tree. Once the simulated particle starts to retrace its trajectory, the tree construction is interrupted and the ideal number of $L$ *leapfrog steps* is defined as $2^j$ in time $j-1$ from the beginning of the retrogression of the trajectory. So the simulated particle never "turns around" so "No U-turn". For more details on the algorithm and how it relates to Hamiltonian dynamics see Hoffman & Gelman (2011).

# ## References

# Betancourt, M. (2017, January 9). A Conceptual Introduction to Hamiltonian Monte Carlo. Retrieved November 6, 2019, from http://arxiv.org/abs/1701.02434
#
# Brooks, S., Gelman, A., Jones, G., & Meng, X.-L. (2011). Handbook of Markov Chain Monte Carlo. Retrieved from https://books.google.com?id=qfRsAIKZ4rIC
#
# Brooks, S. P., & Gelman, A. (1998). General Methods for Monitoring Convergence of Iterative Simulations. Journal of Computational and Graphical Statistics, 7(4), 434–455. https://doi.org/10.1080/10618600.1998.10474787
#
# Casella, G., & George, E. I. (1992). Explaining the gibbs sampler. The American Statistician, 46(3), 167–174. https://doi.org/10.1080/00031305.1992.10475878
#
# Chib, S., & Greenberg, E. (1995). Understanding the Metropolis-Hastings Algorithm. The American Statistician, 49(4), 327–335. https://doi.org/10.1080/00031305.1995.10476177
#
# Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). Hybrid Monte Carlo. Physics Letters B, 195(2), 216–222. https://doi.org/10.1016/0370-2693(87)91197-X
#
# Eckhardt, R. (1987). Stan Ulam, John von Neumann, and the Monte Carlo Method. Los Alamos Science, 15(30), 131–136.
#
# Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in Bayesian workflow. Journal of the Royal Statistical Society: Series A (Statistics in Society), 182(2), 389–402. https://doi.org/10.1111/rssa.12378
#
# Gelman, A. (1992). Iterative and Non-Iterative Simulation Algorithms. Computing Science and Statistics (Interface Proceedings), 24, 457–511. PROCEEDINGS PUBLISHED BY VARIOUS PUBLISHERS.
#
# Gelman, A. (2008). The folk theorem of statistical computing. Retrieved from https://statmodeling.stat.columbia.edu/2008/05/13/the_folk_theore/
#
# Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013a). Basics of Markov Chain Simulation. In Bayesian Data Analysis. Chapman and Hall/CRC.
#
# Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013b). Bayesian Data Analysis. Chapman and Hall/CRC.
#
# Gelman, A., & Rubin, D. B. (1992). Inference from Iterative Simulation Using Multiple Sequences. Statistical Science, 7(4), 457–472. https://doi.org/10.1214/ss/1177011136
#
# Geman, S., & Geman, D. (1984). Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-6(6), 721–741. https://doi.org/10.1109/TPAMI.1984.4767596
#
# Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57(1), 97–109. https://doi.org/10.1093/biomet/57.1.97
#
# Hoffman, M. D., & Gelman, A. (2011). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593–1623. Retrieved from http://arxiv.org/abs/1111.4246
#
# Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of State Calculations by Fast Computing Machines. The Journal of Chemical Physics, 21(6), 1087–1092. https://doi.org/10.1063/1.1699114
#
# Neal, Radford M. (1994). An Improved Acceptance Procedure for the Hybrid Monte Carlo Algorithm. Journal of Computational Physics, 111(1), 194–203. https://doi.org/10.1006/jcph.1994.1054
#
# Neal, Radford M. (2003). Slice Sampling. The Annals of Statistics, 31(3), 705–741. Retrieved from https://www.jstor.org/stable/3448413
#
# Neal, Radford M. (2011). MCMC using Hamiltonian dynamics. In S. Brooks, A. Gelman, G. L. Jones, & X.-L. Meng (Eds.), Handbook of markov chain monte carlo.
#
# Revels, J., Lubin, M., & Papamarkou, T. (2016). Forward-Mode Automatic Differentiation in Julia. ArXiv:1607.07892 [Cs]. http://arxiv.org/abs/1607.07892
#
# Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. Annals of Applied Probability, 7(1), 110–120. https://doi.org/10.1214/aoap/1034625254
#
# Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-Normalization, Folding, and Localization: An Improved Rˆ for Assessing Convergence of MCMC. Bayesian Analysis, 1(1), 1–28. https://doi.org/10.1214/20-BA1221
