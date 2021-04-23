# # Markov Chain Monte Carlo (MCMC)

# The main computational barrier for Bayesian statistics is the denominator $P(\text{data})$ of the Bayes formula:

# $$ P(\theta \mid \text{data})=\frac{P(\theta) \cdot P(\text{data} \mid \theta)}{P(\text{data})} \label{bayes} $$

# In discrete cases we can turn the denominator into a sum of all parameters using the chain rule of probability:

# $$ P(A,B\midC)=P(A\midB,C) \times P(B\midC) \label{chainrule} $$

# This is also called marginalization:

# $$ P(\text{data})=\sum_{\theta} P(\text{data} \mid \theta) \times P(\theta) \label{discretemarginalization} $$

# However, in the continuous cases the denominator $P(\text{data})$ becomes a very large and complicated integral to calculate:

# $$ P(\text{data})=\int_{\theta} P(\text{data} \mid \theta) \times P(\theta)d \theta \label{continuousmarginalization} $$

# In many cases this integral becomes *intractable* (incalculable) and therefore we must find other ways to calculate
# the posterior probability $P(\theta \mid \ text {data})$ in \eqref{bayes} without using the denominator $P(\text{data})$.

# ## What is the denominator $P(\text{data})$ for?

# To normalize the posterior in order to make it a valid probability distribution. This means that the sum of all probabilities
# of the possible events in the probability distribution must be equal to 1:

# - in the case of discrete probability distribution:  $\sum_{\theta} P(\theta \mid \text{data}) = 1$
# - in the case of continuous probability distribution: $\int_{\theta} P(\theta \mid \text{data})d \theta = 1$

# ## What if we remove this denominator?

# When we remove the denominator $(\text{data})$ we have that the posterior $P(\theta \mid \text{data})$ is **proportional** to the
# prior multiplied by the likelihood $P(\theta) \cdot P(\text{data} \mid \theta)$[^propto]:

# $$ P(\theta \mid \text{data}) \propto P(\theta) \cdot P(\text{data} \mid \theta) \label{proptobayes} $$

# This [YouTube video](https://youtu.be/8FbqSVFzmoY) explains the denominator problem very well, see below:

# ~~~
# <style>.embed-container { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; } .embed-container iframe, .embed-container object, .embed-container embed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }</style><div class='embed-container'><iframe src='https://www.youtube.com/embed/8FbqSVFzmoY' frameborder='0' allowfullscreen></iframe></div>
# ~~~


# ## Footnotes
# [^propto]: the symbol $\propto$ (`\propto`) should be read as "proportional to".

# ## References
