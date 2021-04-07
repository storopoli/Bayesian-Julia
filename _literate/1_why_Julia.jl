# # Why Julia?

# I'm going to use $\LaTeX$ and **Julia**.

# ## Here's some maths

# $$ f(x,y) = x^{x} + \frac{y}{x} $$

# And some code

f(x,y) = x^x + y / x

f(2,3)

# Also more math but now labbelled
# $$ e^{i \pi} + 1 = 0 \label{euler} $$

# Then, we can reference the equation as Equation \eqref{euler}.

# ## Testing Plots

using Plots
plot(Plots.fakedata(50, 5), w=3);
savefig(joinpath(@OUTPUT, "test.svg")) # hide

# \fig{test}

# \fig{bayes-meme}
