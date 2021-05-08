# # Multilevel Models (a.k.a. Hierarchical Models)

# Bayesian hierarchical models (also called multilevel models) are a statistical model written at *multiple* levels
# (hierarchical form) that estimates the parameters of the posterior distribution using the Bayesian approach.
# The sub-models combine to form the hierarchical model, and Bayes' theorem is used to integrate them with the
# observed data and to account for all the uncertainty that is present. The result of this integration is the
# posterior distribution, also known as an updated probability estimate, as additional evidence of the likelihood
# function is integrated together with the prior distribution of the parameters.

# Hierarchical modeling is used when information is available at several different levels of observation units.
# The hierarchical form of analysis and organization helps to understand multiparameter problems and also plays
# an important role in the development of computational strategies.

# Hierarchical models are mathematical statements that involve several parameters, so that the estimates of some parameters
# depend significantly on the values of other parameters. The figure below shows a hierarchical model in which there is a
# $\phi$ hyperparameter that parameterizes the parameters $\theta_1, \theta_2, \dots, \theta_N$ that are finally used to
# infer the posterior density of some variable of interest $\mathbf{y} = y_1, y_2, \dots, y_N$.

# ![Bayesian Workflow](/pages/images/hierarchical.png)
#
# \center{*Hierarchical Model*} \\

# ## When to use Multilevel Models?

# Multilevel models are particularly suitable for research projects where participant data is organized at more than one level, *i.e.* nested data.
# Units of analysis are usually individuals (at a lower level) that are nested in contextual/aggregate units (at a higher level).
# An example is when we are measuring the performance of individuals and we have additional information about belonging to different
# groups such as sex, age group, hierarchical level, educational level or housing status.

# There is a main assumption that cannot be violated in multilevel models which is **exchangeability** (de Finetti, 1974; Nau, 2001).
# Yes, this is the same assumption that we discussed in [2. **What is Bayesian Statistics?**](/pages/2_bayes_stats/).
# This assumption assumes that groups are exchangeable. The figure below shows a graphical representation of the exchangeability.
# The groups shown as "cups" that contain observations shown as "balls". If in the model's inferences, this assumption is violated,
# then multilevel models are not appropriate. This means that, since there is no theoretical justification to support exchangeability,
# the inferences of the multilevel model are not robust and the model can suffer from several pathologies and should not be used for any
# scientific or applied analysis.

# ![Bayesian Workflow](/pages/images/exchangeability-1.png)
# ![Bayesian Workflow](/pages/images/exchangeability-2.png)
#
# \center{*Exchangeability -- Images from [Michael Betancourt](https://betanalpha.github.io/)*} \\


# ## References
#
# Boatwright, P., McCulloch, R., & Rossi, P. (1999). Account-level modeling for trade promotion: An application of a constrained parameter hierarchical model. Journal of the American Statistical Association, 94(448), 1063–1073.
#
# de Finetti, B. (1974). Theory of Probability (Volume 1). New York: John Wiley & Sons.
#
# Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis. Chapman and Hall/CRC.
#
# Lewandowski, D., Kurowicka, D., & Joe, H. (2009). Generating random correlation matrices based on vines and extended onion method. Journal of Multivariate Analysis, 100(9), 1989–2001.
#
# Nau, R. F. (2001). De Finetti was Right: Probability Does Not Exist. Theory and Decision, 51(2), 89–124. https://doi.org/10.1023/A:1015525808214
