# # Computational Tricks with Turing (Non-Centered Parametrization and QR Decomposition)

# There are some computational tricks that we can employ with Turing.
# I will cover here two computational tricks:

# 1. **QR Decomposition**
# 2. **Non-Centered Parametrization**

# ## QR Decomposition

# Back in "Linear Algebra 101" we've learned that any matrix (even retangular ones) can be factored
# into the product of two matrices:

# * $\mathbf{Q}$: an orthogonal matrix (its columns are orthogonal unit vectors meaning $\mathbf{Q}^T = \mathbf{Q}^{-1})$.
# * $\mathbf{R}$: an upper triangular matrix.

# This is commonly known as the [**QR Decomposition**](https://en.wikipedia.org/wiki/QR_decomposition):

# $$ \mathbf{A} = \mathbf{Q} \cdot \mathbf{R} $$

# Let me show you an example with a random matrix $\mathbf{A} \in \mathbb{R}^{3 \times 2}$:

A = rand(3, 2)

# Now let's factor `A` using `LinearAlgebra`'s `qr()` function:

using LinearAlgebra:qr, I
Q, R = qr(A)

# Notice that `qr()` produced a tuple containing two matrices `Q` and `R`. `Q` is a 3x3 orthogonal matrix.
# And `R` is a 2x2 upper triangular matrix.
# So that $\mathbf{Q}^T = \mathbf{Q}^{-1}$ (the transpose is equal the inverse):

Matrix(Q') ≈ Matrix(Q^-1)

# Also note that $\mathbf{Q}^T \cdot \mathbf{Q}^{-1} = \mathbf{I}$ (identity matrix):

Q' * Q ≈ I(3)

# Let's go back to the example in [6. **Bayesian Linear Regression**](/pages/6_linear_reg/)

using Turing

# ```julia
# Summary Statistics
#   parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec
#       Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64
#
#            α   21.5724    8.7260     0.0976    0.1646   2947.6610    1.0041      212.8275
#         β[1]    2.0223    1.8276     0.0204    0.0291   3760.2863    1.0006      271.5008
#         β[2]    0.5802    0.0589     0.0007    0.0009   4363.1476    1.0019      315.0287
#         β[3]    0.2469    0.3081     0.0034    0.0051   3393.1174    1.0016      244.9904
#            σ   17.8753    0.6013     0.0067    0.0080   5809.2999    1.0004      419.4440
# ```

# Benchmarking

using BenchmarkTools, Random

Random.seed!(123)

data = rand(Normal());

# ```julia
# @btime sample(linear_reg($data), NUTS(), 2_000)
# ```


# Takes XXX in my machine

# Using `FillArrays.jl`

# ```julia
# x ~ MvNormal(Fill(m, length(x)), 0.2)
# ```

# Now the efficient stuff

using LazyArrays
lazyarray(f, x) = LazyArray(Base.broadcasted(f, x))

# Model


# Benchmarking

