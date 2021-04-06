# # Why Julia?

# I'm going to use $\LaTeX$ and **Julia**.

# ## Here's some maths

# $ f(x,y) = x^{x} + \frac{y}{x} $

### And some code

f(x,y) = x^x + y / x

f(2,3)

# ## Testing Plots

using Plots
plot(Plots.fakedata(50, 5), w=3);
savefig(joinpath(@OUTPUT, "test.svg")) # hide

# \figalt{Fake Data}{test}
