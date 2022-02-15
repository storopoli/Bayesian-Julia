# # Why Julia?

# [Julia](https://www.julialang.org) (Bezanson, Edelman, Karpinski & Shah, 2017) is a relatively new language, first released in 2012, aims to be both **high-level** and **fast**.
# Julia is a fast dynamic-typed language that just-in-time (JIT)
# compiles into native code using LLVM. It ["runs like C but reads like Python"](https://www.nature.com/articles/d41586-019-02310-3),
# meaning that is *blazing* fast, easy to prototype and read/write code.
# It is **multi-paradigm**, combining features of imperative, functional, and object-oriented programming.

# **Why was Julia created?** Definitely read this now impressively
# [old post by Julia founders](https://julialang.org/blog/2012/02/why-we-created-julia/).
# Here is a clarifying quote:

# > We want the speed of C with the dynamism of Ruby. We want a language that's homoiconic, with true macros like Lisp, but with obvious,
# > familiar mathematical notation like Matlab. We want something as usable for general programming as Python, as easy for statistics as R,
# > as natural for string processing as Perl, as powerful for linear algebra as Matlab, as good at gluing programs together as the shell.
# > Something that is dirt simple to learn, yet keeps the most serious hackers happy. We want it interactive and we want it compiled.

# **Why this needs to be an extra language?** Why cannot Python (or R) be made that fast for instance?
# See the official compact answer to this in the [Julia manual FAQ](https://docs.julialang.org/en/v1/manual/faq/#Why-don't-you-compile-Matlab/Python/R/%E2%80%A6-code-to-Julia?):

# > The basic issue is that there is nothing special about Julia's compiler: we use a commonplace compiler (LLVM) with no
# > "secret sauce" that other language developers don't know about.
# > Julia's performance advantage derives almost entirely from its front-end: its language semantics allow a well-written Julia program
# > to give more opportunities to the compiler to generate efficient code and memory layouts. If you tried to compile Matlab or
# > Python code to Julia, our compiler would be limited by the semantics of Matlab or Python to producing code no better than that of
# > existing compilers for those languages (and probably worse).
# >
# > Julia's advantage is that good performance is not limited to a small subset of "built-in" types and operations,
# > and one can write high-level type-generic code that works on arbitrary user-defined types while remaining fast and memory-efficient.
# > Types in languages like Python simply don't provide enough information to the compiler for similar capabilities, so as soon as you used
# > those languages as a Julia front-end you would be stuck.

# These are the "official" answers from the Julia community. Now let me share with you my opinion.
# From my point-of-view Julia has **three main features that makes it a unique language to work with**, specially in scientific computing:

# * **Speed**
# * **Ease of Use**
# * **Multiple Dispatch**

# Now let's dive into each one of those three features.

# ## Speed

# Yes, Julia is **fast**. **Very fast!** It was made for speed from the drawing board. It bypass any sort of intermediate representation and translate
# code into machine native code using LLVM compiler. Comparing this with R, that uses either FORTRAN or C, or Python, that uses CPython;
# and you'll clearly see that Julia has a major speed advantage over other languages that are common in data science and statistics.
# Julia exposes the machine code to LLVM's compiler which in turn can optimize code as it wishes, like a good compiler such as LLVM excels in.

# One notable example: NASA uses Julia to analyze the
# "[Largest Batch of Earth-Sized Planets Ever Found](https://exoplanets.nasa.gov/news/1669/seven-rocky-trappist-1-planets-may-be-made-of-similar-stuff/)".
# Also, you can find [benchmarks](https://julialang.org/benchmarks/) for a range of common code patterns, such as function calls, string
# parsing, sorting, numerical loops, random number generation, recursion, and array operations using Julia and also several
# other languages such as C, Rust, Go, JavaScript, R, Python, Fortran and Java. The figure below was taken from
# [Julia's website](https://julialang.org/benchmarks/). As you can see Julia is **indeed** fast:

# ![Common Benchmarks](/pages/images/benchmarks.svg)
#
# \center{*Common Benchmarks*} \\

# Let me demonstrate how fast Julia is. Here is a simple "groupby" operation using random stuff to emulate common data analysis
# "split-apply-combine" operations in three languages[^updatedversion] :
#
# * Julia: using [`DataFrames.jl`](https://dataframes.juliadata.org/stable/) - 0.4ms
# * Python: using `Pandas` and `NumPy` - 1.76ms
# * R: using `{dplyr}` - 3.22ms

# Here is Julia:

# ```julia
# using Random, StatsBase, DataFrames, BenchmarkTools, Chain
# Random.seed!(123)
#
# n = 10_000
#
# df = DataFrame(
#     x=sample(["A", "B", "C", "D"], n, replace=true),
#     y=rand(n),
#     z=randn(n),
# )
#
# @btime @chain $df begin  # passing `df` as reference so the compiler cannot optimize
#     groupby(:x)
#     combine(:y => median, :z => mean)
# end
# ```

# Here is Python:

# ```python
# import pandas as pd
# import numpy as np
#
# n = 10000
#
# df = pd.DataFrame({'x': np.random.choice(['A', 'B', 'C', 'D'], n, replace=True),
#                    'y': np.random.randn(n),
#                    'z': np.random.rand(n)})
#
# %timeit df.groupby('x').agg({'y': 'median', 'z': 'mean'})
# ```

# Here is R:

# ```r
# library(dplyr)
#
# n <- 10e3
# df <- tibble(
#     x = sample(c("A", "B", "C", "D"), n, replace = TRUE),
#     y = runif(n),
#     z = rnorm(n)
# )
#
# bench::mark(
#     df %>%
#         group_by(x) %>%
#         summarize(
#             median(y),
#             mean(z)
#         )
# )
# ```

# So clearly **Julia is the winner here**, being **4x faster than Python** and almost **10x faster than R**. Also note that `Pandas`
# (along with `NumPy`) and `{dplyr}` are all written in C or C++. Additionally, I didn't let Julia cheat by allowing the compiler
# optimize for `df` by passing a reference `$df`. So, I guess this is a fair comparison.

# ## Ease of Use

# What is most striking that Julia can be as fast as C (and faster than Java in some applications) while **having a very simple and
# intelligible syntax**. This feature along with its speed is what Julia creators denote as **"the two language problem"** that Julia
# address. The **"two language problem" is a very typical situation in scientific computing** where a researcher or computer scientist
# devises an algorithm or a solution that he or she prototypes in an easy to code language (like Python) and, if it works, he or she
# would code in a fast language that is not easy to code (C or FORTRAN). Thus, we have two languages involved in the process of
# of developing a new solution. One which is easy to prototype but is not suited for implementation (mostly due to  being slow).
# And another one which is not so easy to code (and, consequently, not easy to prototype) but suited for implementation
# (mostly because it is fast). Julia comes to **eliminate such situations** by being the **same language** that you **prototype** (ease of use)
# and **implement the solution** (speed).

# Also, Julia lets you use **unicode characters as variables or parameters**. This means no more using `sigma` or `sigma_i`,
# and instead just use `σ` or `σᵢ` as you would in mathematical notation. When you see code for an algorithm or for a
# mathematical equation you see a **one-to-one relation to code and math**. This is a **powerful** feature.

# I think that the "two language problem" and the one-to-one code and math relation are best described by
# one of the creators of Julia, Alan Edelman, in a **TED Talk** (see the video below):

# ~~~
# <style>.embed-container { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; } .embed-container iframe, .embed-container object, .embed-container embed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }</style><div class='embed-container'><iframe src='https://www.youtube.com/embed/qGW0GT1rCvs' frameborder='0' allowfullscreen></iframe></div>
# ~~~

# I will try to exemplify what would be the "two language problem" by showing you how I would code a simple
# [**Metropolis algorithm**](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
# for a **bivariate normal distribution**. I would mostly prototype it in a dynamically-typed language such as R or Python.
# Then, deploy the algorithm using a fast but hard to code language such as C++. This is exactly what I'll do now.
# The algorithm will be coded in **Julia**, **R**, **C++** and [**`Stan`**](https://mc-stan.org). There are two caveats.
# First, I am coding the **original 1950s Metropolis version**, not the **1970s Metropolis-Hastings**, which implies
# **symmetrical proposal distributions** just for the sake of the example.
# Second, the proposals are based on a **uniform distribution** on the current proposal values of the proposal values ± a certain `width`.
#
# Let's start with **Julia** which uses the [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/) package for
# its probabilistic distributions along with `logpdf()` defined methods for all of the distributions.

# ```julia
# using Distributions
# function metropolis(S::Int64, width::Float64, ρ::Float64;
#                     μ_x::Float64=0.0, μ_y::Float64=0.0,
#                     σ_x::Float64=1.0, σ_y::Float64=1.0)
#     binormal = MvNormal([μ_x; μ_y], [σ_x ρ; ρ σ_y]);
#     draws = Matrix{Float64}(undef, S, 2);
#     x = randn(); y = randn();
#     accepted = 0::Int64;
#     for s in 1:S
#         x_ = rand(Uniform(x - width, x + width));
#         y_ = rand(Uniform(y - width, y + width));
#         r = exp(logpdf(binormal, [x_, y_]) - logpdf(binormal, [x, y]));
#
#         if r > rand(Uniform())
#             x = x_;
#             y = y_;
#             accepted += 1;
#         end
#         @inbounds draws[s, :] = [x y];
#     end
#     println("Acceptance rate is $(accepted / S)")
#     return draws
# end
# ```

# Now let's go to the **R version** (from now on no more fancy names like `μ` or `σ` 😭).
# Since this is a bivariate normal I am using the package [`{mnormt}`](https://cran.r-project.org/web/packages/mnormt/index.html)
# which allows for very fast (FORTRAN code) computation of multivariate normal distributions' pdf and logpdf.

# ```r
# metropolis <- function(S, width,
#                        mu_X = 0, mu_Y = 0,
#                        sigma_X = 1, sigma_Y = 1,
#                        rho) {
#    Sigma <- diag(2)
#    Sigma[1, 2] <- rho
#    Sigma[2, 1] <- rho
#    draws <- matrix(nrow = S, ncol = 2)
#    x <- rnorm(1)
#    y <- rnorm(1)
#    accepted <- 0
#    for (s in 1:S) {
#       x_ <- runif(1, x - width, x + width)
#       y_ <- runif(1, y - width, y + width)
#       r <- exp(mnormt::dmnorm(c(x_, y_), mean = c(mu_X, mu_Y), varcov = Sigma, log = TRUE) -
#                         mnormt::dmnorm(c(x, y), mean = c(mu_X, mu_Y), varcov = Sigma, log = TRUE))
#       if (r > runif(1, 0, 1)) {
#         x <- x_
#         y <- y_
#         accepted <- accepted + 1
#       }
#       draws[s, 1] <- x
#       draws[s, 2] <- y
#    }
#    print(paste0("Acceptance rate is ", accepted / S))
#    return(draws)
# }
# ```

# Now **C++**. Here I am using the [`Eigen`](https://eigen.tuxfamily.org/) library. Note that, since C++ is a very powerful language
# to be used as "close to the metal" as possible, I don't have any
# convenient predefined multivariate normal to use. So I will have to create this from zero[^mvnimplem]. Ok, be **ready**!
# This is a mouthful:

# ```cpp
# #include <Eigen/Eigen>
# #include <cmath>
# #include <iostream>
# #include <random>
#
# using std::cout;
# std::random_device rd{};
# std::mt19937 gen{rd()};
#
# // Random Number Generator Stuff
# double random_normal(double mean = 0, double std = 1) {
#   std::normal_distribution<double> d{mean, std};
#   return d(gen);
# };
#
# double random_unif(double min = 0, double max = 1) {
#   std::uniform_real_distribution<double> d{min, max};
#   return d(gen);
# };
#
# // Multivariate Normal
# struct Mvn {
#   Mvn(const Eigen::VectorXd &mu, const Eigen::MatrixXd &s)
#       : mean(mu), sigma(s) {}
#   double pdf(const Eigen::VectorXd &x) const;
#   double lpdf(const Eigen::VectorXd &x) const;
#   Eigen::VectorXd mean;
#   Eigen::MatrixXd sigma;
# };
#
# double Mvn::pdf(const Eigen::VectorXd &x) const {
#   double n = x.rows();
#   double sqrt2pi = std::sqrt(2 * M_PI);
#   double quadform = (x - mean).transpose() * sigma.inverse() * (x - mean);
#   double norm = std::pow(sqrt2pi, -n) * std::pow(sigma.determinant(), -0.5);
#
#   return norm * exp(-0.5 * quadform);
# }
#
# double Mvn::lpdf(const Eigen::VectorXd &x) const {
#   double n = x.rows();
#   double sqrt2pi = std::sqrt(2 * M_PI);
#   double quadform = (x - mean).transpose() * sigma.inverse() * (x - mean);
#   double norm = std::pow(sqrt2pi, -n) * std::pow(sigma.determinant(), -0.5);
#
#   return log(norm) + (-0.5 * quadform);
# }
#
# Eigen::MatrixXd metropolis(int S, double width, double mu_X = 0,
#                                  double mu_Y = 0, double sigma_X = 1,
#                                  double sigma_Y = 1, double rho = 0.8) {
#   Eigen::MatrixXd sigma(2, 2);
#   sigma << sigma_X, rho, rho, sigma_Y;
#   Eigen::VectorXd mean(2);
#   mean << mu_X, mu_Y;
#   Mvn binormal(mean, sigma);
#
#   Eigen::MatrixXd out(S, 2);
#   double x = random_normal();
#   double y = random_normal();
#   double accepted = 0;
#   for (size_t i = 0; i < S - 1; i++) {
#     double xmw = x - width;
#     double xpw = x + width;
#     double ymw = y - width;
#     double ypw = y + width;
#
#     double x_ = random_unif(xmw, xpw);
#     double y_ = random_unif(ymw, ypw);
#
#     double r = std::exp(binormal.lpdf(Eigen::Vector2d(x_, y_)) -
#                         binormal.lpdf(Eigen::Vector2d(x, y)));
#     if (r > random_unif()) {
#       x = x_;
#       y = y_;
#       accepted++;
#     }
#     out(i, 0) = x;
#     out(i, 1) = y;
#   }
#   cout << "Acceptance rate is " << accepted / S << '\n';

#   return out;
# }
# ```

# note that the [PDF for a multivariate normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) is:
#
# $$ \text{PDF}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = (2\pi)^{-{\frac{k}{2}}}\det({\boldsymbol{\Sigma}})^{-{\frac {1}{2}}}e^{-{\frac{1}{2}}(\mathbf{x}-{\boldsymbol{\mu}})^{T}{\boldsymbol{\Sigma }}^{-1}(\mathbf{x} -{\boldsymbol{\mu}})} \label{mvnpdf} , $$
#
# where $\boldsymbol{\mu}$ is a vector of means, $k$ is the number of dimensions, $\boldsymbol{\Sigma}$ is a covariance matrix, $\det$ is the determinant and $\mathbf{x}$
# is a vector of values that the PDF is evaluted for.

# **SPOILER ALERT**: Julia will beat this C++ Eigen implementation by being almost 100x faster. So I will try to *help* C++ beat Julia (😂)
# by making a bivariate normal class `BiNormal` in order to avoid the expensive operation of inverting a covariance matrix and computing
# determinants in every logpdf proposal evaluation.
# Also since we are not doing linear algebra computations I've removed Eigen and used C++ STL's `<vector>`:

# ```cpp
# #define M_PI 3.14159265358979323846 /* pi */
#
# // Bivariate Normal
# struct BiNormal {
#   BiNormal(const std::vector<double> &mu, const double &rho)
#       : mean(mu), rho(rho) {}
#   double pdf(const std::vector<double> &x) const;
#   double lpdf(const std::vector<double> &x) const;
#   std::vector<double> mean;
#   double rho;
# };
#
# double BiNormal::pdf(const std::vector<double> &x) const {
#   double x_ = x[0];
#   double y_ = x[1];
#   return std::exp(-((std::pow(x_, 2) - (2 * rho * x_ * y_) + std::pow(y_, 2)) /
#                     (2 * (1 - std::pow(rho, 2))))) /
#          (2 * M_PI * std::sqrt(1 - std::pow(rho, 2)));
# }
#
# double BiNormal::lpdf(const std::vector<double> &x) const {
#   double x_ = x[0];
#   double y_ = x[1];
#   return (-((std::pow(x_, 2) - (2 * rho * x_ * y_) + std::pow(y_, 2))) /
#           (2 * (1 - std::pow(rho, 2)))) -
#          std::log(2) - std::log(M_PI) - log(std::sqrt(1 - std::pow(rho, 2)));
# }
# ```

# This means that I've simplified the PDF [^mathbinormal] from equation \eqref{mvnpdf} into:
#
# $$ \text{PDF}(x, y)= \frac{1}{2 \pi \sqrt{1 - \rho^2 } \sigma_X \sigma_Y} e^{-\frac{\frac{x^{2}}{\sigma_{X}^{2}}-2 \rho-\frac{x y}{\sigma_{X} \sigma_{Y}}+\frac{y^{2}}{\sigma_{Y}^{2}}}{2\left(1-\rho^{2}\right)}} \label{bvnpdf} .$$
#
# Since $\sigma_{X} = \sigma_{Y} = 1$, equation \eqref{bvnpdf} boils down to:
#
# $$ \text{PDF}(x, y)=\frac{1}{2 \pi \sqrt{1 - \rho^2 }} e^{-\frac{x^{2} -2 \rho-x y+y^{2}}{2\left(1-\rho^{2}\right)}} .$$
#
# no more determinants or matrix inversions. Easy-peasy for C++.

# Now let's go to the last, but not least: [`Stan`](https://mc-stan.org) is a probabilistic language for specifying probabilistic
# models (does the same as `Turing.jl` does) and comes also with a very fast C++-based MCMC sampler. `Stan` is a personal favorite
# of mine and I have a [whole graduate course of Bayesian statistics using `Stan`](https://storopoli.io/Estatistica-Bayesiana/).
# Here's the `Stan` implementation:

# ```stan
# functions {
#     real binormal_lpdf(real [] xy, real mu_X, real mu_Y, real sigma_X, real sigma_Y, real rho) {
#     real beta = rho * sigma_Y / sigma_X; real sigma = sigma_Y * sqrt(1 - square(rho));
#     return normal_lpdf(xy[1] | mu_X, sigma_X) +
#            normal_lpdf(xy[2] | mu_Y + beta * (xy[1] - mu_X), sigma);
#   }
#
#   matrix metropolis_rng(int S, real width,
#                         real mu_X, real mu_Y,
#                         real sigma_X, real sigma_Y,
#                         real rho) {
#     matrix[S, 2] out; real x = normal_rng(0, 1); real y = normal_rng(0, 1); real accepted = 0;
#     for (s in 1:S) {
#       real xmw = x - width; real xpw = x + width; real ymw = y - width; real ypw = y + width;
#       real x_ = uniform_rng(xmw, xpw); real y_ = uniform_rng(ymw, ypw);
#       real r = exp(binormal_lpdf({x_, y_} | mu_X, mu_Y, sigma_X, sigma_Y, rho) -
#                             binormal_lpdf({x , y } | mu_X, mu_Y, sigma_X, sigma_Y, rho));
#       if (r > uniform_rng(0, 1)) {
#         x = x_; y = y_; accepted += 1;
#       }
#       out[s, 1] = x;  out[s, 2] = y;
#     }
#     print("Acceptance rate is ", accepted / S);
#     return out;
#   }
# }
# ```

# Wow, that was lot... Not let's go to the results. I've benchmarked R and `Stan` code using `{bench}` and `{rstan}` packages, C++ using `catch2`, Julia using
# `BenchmarkTools.jl`. For all benchmarks the parameters were: `S = 10_000` simulations, `width = 2.75` and `ρ = 0.8`.
# From fastest to slowest:
#
# * `Stan` - 3.6ms
# * Julia - 6.3ms
# * C++ `BiNormal` - 17ms
# * C++ `MvNormal` - 592ms
# * R - 1.35s which means 1350ms

# **Conclusion**: a naïve Julia implementation beats C++
# (while also beating a C++ math-helped faster implementation using bivariate normal PDFs) and gets very close to `Stan`,
# a highly specialized  probabilistic language that compiles and runs on C++ with lots of contributors, funding and development
# time invested.

# Despite being *blazing* fast, Julia also **codes very easily**. You can write and read code without much effort.

# ## Multiple Dispatch

# I think that this is the **real gamechanger of Julia language**: The ability to define **function behavior** across many combinations of argument
# types via [**multiple dispatch**](https://en.wikipedia.org/wiki/Multiple_dispatch). **Multiple dispatch** is a feature
# that allows a function or method to be **dynamically dispatched** based on the run-time (dynamic) type or,
# in the more general case, some other attribute of more than one of its arguments. This is a **generalization of
# single-dispatch polymorphism** where a function or method call is dynamically dispatched based on the derived type of
# the object on which the method has been called. Multiple dispatch routes the dynamic dispatch to the
# implementing function or method using the combined characteristics of one or more arguments.

# Most languages have single-dispatch polymorphism that rely on the first parameter of a method in order to
# determine which method should be called. But what Julia differs is that **multiple parameters are taken into account**.
# This enables multiple definitions of similar functions that have the same initial parameter.
# I think that this is best explained by one of the creators of Julia, Stefan Karpinski, at JuliaCon 2019 (see the video below):

# ~~~
# <style>.embed-container { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; } .embed-container iframe, .embed-container object, .embed-container embed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }</style><div class='embed-container'><iframe src='https://www.youtube.com/embed/kc9HwsxE1OY' frameborder='0' allowfullscreen></iframe></div>
# ~~~
#
# ### Example: Dogs and Cats
#
# I will reproduce Karpinski's example. In the talk, Karpinski designs a structure of classes which are very common in
# object-oriented programming (OOP). In Julia, we don't have classes but we have **structures** (`struct`) that are meant to be
# "structured data": they define the kind of information that is embedded in the structure,
# that is a set of fields (aka "properties" or "attributes" in other languages), and then individual instances (or "objects") can
# be produced each with its own specific values for the fields defined by the structure.
#
# We create an abstract `type` called `Pet`.
# Then, we proceed by creating two derived `struct` from `Pet` that has one field `name` (a `String`).
# These derived `struct` are `Dog` and `Cat`. We also define some methods for what happens in an "encounter" by defining
# a generic function `meets()` and several specific methods of `meets()` that will be multiple dispatched by Julia in runtime
# to define the action that one type `Pet` takes when it meets another `Pet`:

abstract type Pet end
struct Dog <: Pet name::String end
struct Cat <: Pet name::String end

function encounter(a::Pet, b::Pet)
    verb = meets(a, b)
    println("$(a.name) meets $(b.name) and $verb")
end

meets(a::Dog, b::Dog) = "sniffs";
meets(a::Dog, b::Cat) = "chases";
meets(a::Cat, b::Dog) = "hisses";
meets(a::Cat, b::Cat) = "slinks";

# Let's see what happens when we instantiate objects from `Dog` and `Cat` and
# call `encounter` on them in Julia:

fido = Dog("Fido");
rex = Dog("Rex");
whiskers = Cat("Whiskers");
spots = Cat("Spots");

encounter(fido, rex)
encounter(rex, whiskers)
encounter(spots, fido)
encounter(whiskers, spots)

# It works as expected. Now let's translate this to modern C++ as literally as possible.
# Let's define a class `Pet` with a member variable `name` -- in C ++ we can do this. Then we define a base function `meets()`,
# a function `encounter() `for two objects of the type `Pet`, and finally, define derived classes `Dog `and `Cat`
# overload `meets()` for them:
#
# ```cpp
# #include <iostream>
# #include <string>
#
# using std::string;
# using std::cout;
#
# class Pet {
#     public:
#         string name;
# };
#
# string meets(Pet a, Pet b) { return "FALLBACK"; } // If we use `return meets(a, b)` doesn't work
#
# void encounter(Pet a, Pet b) {
#     string verb = meets(a, b);
#     cout << a.name << " meets "
#          << b. name << " and " << verb << '\n';
# }
#
# class Cat : public Pet {};
# class Dog : public Pet {};
#
# string meets(Dog a, Dog b) { return "sniffs"; }
# string meets(Dog a, Cat b) { return "chases"; }
# string meets(Cat a, Dog b) { return "hisses"; }
# string meets(Cat a, Cat b) { return "slinks"; }
# ```

# Now we add a `main()` function to the C++ script:
#
# ```cpp
# int main() {
#     Dog fido;      fido.name     = "Fido";
#     Dog rex;       rex.name      = "Rex";
#     Cat whiskers;  whiskers.name = "Whiskers";
#     Cat spots;     spots.name    = "Spots";
#
#     encounter(fido, rex);
#     encounter(rex, whiskers);
#     encounter(spots, fido);
#     encounter(whiskers, spots);
#
#     return 0;
# }
# ```

# And this is what we get:
#
# ```bash
# g++ main.cpp && ./a.out
#
# Fido meets Rex and FALLBACK
# Rex meets Whiskers and FALLBACK
# Spots meets Fido and FALLBACK
# Whiskers meets Spots and FALLBACK
# ```
#
# Doesn't work... 🤷🏼
#
# ### Example: One-hot Vector
#
# Now let's change to another nice example of creating a [one-hot vector](https://en.wikipedia.org/wiki/One-hot).
# One-hot vector is a vector of integers in which all indices are zero (0) except for one single index that is one (1).
# In machine learning, one-hot encoding is a frequently used method to deal with categorical data. Because many machine
# learning models need their input variables to be numeric, categorical variables need to be transformed in the pre-processing part.
# The example below is heavily inspired by a [post from Vasily Pisarev](https://habr.com/ru/post/468609/)[^onehotpost].
#
# How we would represent one-hot vectors in Julia? Simple: we create a new type `OneHotVector` in Julia using the `struct` keyword
# and define two fields `len` and `ind`, which represents the `OneHotVector` length and which index is the entry 1
# (*i.e.* which index is "hot"). Then, we define new methods for the `Base` functions `size()` and `getindex()` for our newly defined
# `OneHotVector`.

import Base: size, getindex

struct OneHotVector <: AbstractVector{Int}
    len::Int
    ind::Int
end

size(v::OneHotVector) = (v.len,)

getindex(v::OneHotVector, i::Integer) = Int(i == v.ind)

# Since `OneHotVector` is a `struct` derived from `AbstractVector` we can use all of the methods previously defined for
# `AbstractVector` and it simply works right off the bat. Here we are constructing an `Array` with a list comprehension:

onehot = [OneHotVector(3, rand(1:3)) for _ in 1:4]

# Now I define a new function `inner_sum()` that is basically a recursive dot product with a summation.
# Here A -- this is something matrix-like (although I did not indicate the types, and you can guess something only by the name),
# and `vs` is a vector of some vector-like elements. The function proceeds by taking the dot product of the "matrix"
# with all vector-like elements of `vs` and returning the accumulated values.
# This is all given generic definition without specifying any types.
# Generic programming here consists in this very function call `inner()` in a loop.

using LinearAlgebra

function inner_sum(A, vs)
    t = zero(eltype(A))
    for v in vs
        t += inner(v, A, v) # multiple dispatch!
    end
    return t
end

inner(v, A, w) = dot(v, A * w) # very general definition

# So, "look mom, it works":

A = rand(3, 3)
vs = [rand(3) for _ in 1:4]
inner_sum(A, vs)

# Since `OneHotVector` is a subtype of `AbstractVector`:

supertype(OneHotVector)

# We can use `inner_sum` and it will do what it is supposed to do:

inner_sum(A, onehot)

# But this default implementation is **slow**:

using BenchmarkTools

@btime inner_sum($A, $onehot);

# We can greatly optimize this procedure. Now let's redefine matrix multiplication by `OneHotVector`
# with a simple column selection. We do this by defining a new method of the `*` function (multiplier function)
# of `Base` Julia. Additionally we also create a new optimized method of `inner()` for dealing with `OneHotVector`:

import Base:*

*(A::AbstractMatrix, v::OneHotVector) = A[:, v.ind]
inner(v::OneHotVector, A, w::OneHotVector) = A[v.ind, w.ind]

# That's it! Simple, huh? Now let's benchmark:

@btime inner_sum($A, $onehot);

# **Huge gains** of speed! 🚀

# ## Julia: the right approach

# Here are some more thoughts on why I believe Julia is the right approach to scientific computation.

# Below is a very opinionated image that divides the scientific computing languages that we've spoken so far in a 2x2
# diagram with two axes: *Slow-Fast* and *Easy-Hard*. I've put C++ and FORTRAN in the hard and fast quadrant. R and Python goes into
# the easy and slow quadrant. Julia is the only language in the easy and fast quadrant. I don't know any language that would want
# to be hard and slow, so this quadrant is empty.

# ![Scientific Computing Language Comparisons](/pages/images/language_comparisons.svg)
#
# \center{*Scientific Computing Language Comparisons*} \\

# What I want to say with this image is that if you want to **code fast and easy** use Julia.

# One other thing to note that I find quite astonishing is that Julia packages are all written in Julia. This does not happen in other scientific
# computing languages. For example, the whole `{tidyverse}` ecosystem of R packages are based on C++. `NumPy` and `SciPy` are a mix
# of FORTRAN and C. `Scikit-Learn` is also coded in C.

# See the figure below where I compare the GitHub's "Languages" stack bar of
# [`PyTorch`](https://github.com/pytorch/pytorch), [`TensorFlow`](https://github.com/tensorflow/tensorflow) and
# [`Flux.jl`](https://github.com/FluxML/Flux.jl)(Julia's Deep Learning package). This figure I would call *"Python my a**!"* 😂:

# ![Python my ass](/pages/images/ML_code_breakdown.svg)
#
# \center{*Python my a**!*} \\

# \note{On the other hand, language *interoperability* is extremely useful:
# we want to exploit existing high-quality code in other languages from Julia (and vice versa)!
# Julia community have worked hard on this, from the built-in intrinsic Julia `ccall` function (to call C and Fortran libraries)
# to [JuliaInterop](https://github.com/JuliaInterop)[^interop] packages that connect Julia to Python, R, Matlab, C++, and more.}

# Another example comes from a Julia podcast that unfortunately I cannot recollect either what podcast was nor who was being interviewed.
# While being asked about how he joined the Julia bandwagon, he replied something in the likes:
#
# > *"Well, I was doing some crazy calculation using a library that was a wrapper to an algorithm
# > coded in FORTRAN and I was trying to get help with a bug. I opened an issue and after 2 weeks of no reply
# > I've dived into the FORTRAN code (despite having no knowledge of FORTRAN). There I saw a comment from the original author
# > describing exactly the same bug that I was experiencing and saying that he would fix this in the future. The comment
# > was dated from 1992. At the same time a colleague of mine suggested that I could try to code the algorithm in some
# > new language called Julia.
# > I thought 'me?! code an algorithm?!'. So, I coded the algorithm in Julia and it was faster than the FORTRAN implementation
# > and also without the evil bug. One thing to note that it was really easy to code the algorithm in Julia."*
#
# Having stuff in different language and wrappers can hinder further research as you can see from this example.

# As you saw from the [Karpinski's talk](https://youtu.be/kc9HwsxE1OY) above, **multiple dispatch empower users to define their
# own types (if necessary)** and also allows them to **extend functions and types from other users** to their own special use.
# This results in an ecosystem that stimulates code sharing and code reuse in scientific computing that is
# unmatched. For instance, if I plug a differential equation from [`DifferentialEquations.jl`](https://diffeq.sciml.ai/) into
# a [`Turing.jl`](https://turing.ml/) model I get a Bayesian stochastic differential equation model, *e.g.* **Bayesian SIR model
# for infectious disease**. If I plug a [`Flux.jl`](https://fluxml.ai/) neural network into a [`Turing.jl`](https://turing.ml/)
# model I get a **Bayesian neural network**! When I saw this type of code sharing I was blown away (and I still am).
#
# \note{This is the **true power** of a scientific computing language like Julia. It brings so much **power** and **flexibility** to the
# user and allows different ways of **sharing**, **contributing**, **extending**, **mixing** and **implementing** code and science.
# I hope this short dive into Julia has somehow sent you **towards** Julia.}
#
# ## Footnotes
#
# [^updatedversion]: please note that I've used updated versions for all languages and packages as of April, 2021. `DataFrames.jl` version 1.0.1, `Pandas` version 1.2.4, `NumPy` version 1.20.2, `{dplyr}` version 1.0.5. We did not cover R's `{data.table}` here. Further benchmarking information is available for example here: [Tabular data benchmarking](https://h2oai.github.io/db-benchmark/)
# [^mvnimplem]: which of course I did not. The `Mvn` class is inspired by [Iason Sarantopoulos' implementation](http://blog.sarantop.com/notes/mvn).
# [^mathbinormal]: you can find all the math [here](http://www.athenasc.com/Bivariate-Normal.pdf).
# [^onehotpost]: the post in Russian, I've "Google Translated" it to English.
# [^interop]: Julia has a lot of interoperability between languages. Check out: [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl) and [`JuliaPy`](https://github.com/JuliaPy) for Python; [`RCall.jl`](https://juliainterop.github.io/RCall.jl/stable/) for Java; [`Cxx.jl`](https://juliainterop.github.io/Cxx.jl/stable/) and [`CxxWrap.jl`](https://github.com/JuliaInterop/CxxWrap.jl) for C++; [`Clang.jl`](https://github.com/JuliaInterop/Clang.jl) for libclang and C; [`ObjectiveC.jl`](https://github.com/JuliaInterop/ObjectiveC.jl) for Objective-C; [`JavaCall.jl`](https://juliainterop.github.io/JavaCall.jl/) for Java; [`MATLAB.jl`](https://github.com/JuliaInterop/MATLAB.jl) for MATLAB; [`MathLink.jl`](https://github.com/JuliaInterop/MathLink.jl) for Mathematica/Wolfram Engine; [`OctCall.jl`](https://github.com/JuliaInterop/OctCall.jl) for GNU Octave; and [`ZMQ.jl`](https://juliainterop.github.io/ZMQ.jl/stable/) for ZeroMQ.

# ## References

# Bezanson, J., Edelman, A., Karpinski, S., & Shah, V. B. (2017). Julia: A fresh approach to numerical computing. SIAM Review, 59(1), 65–98.
