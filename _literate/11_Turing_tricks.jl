# # Computational Tricks with Turing (Non-Centered Parametrization and QR Decomposition)

# Let's go back to the example in 6_linear_reg.jl

using Turing


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

