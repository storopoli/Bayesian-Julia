# # What is Bayesian Statistics?

# **Bayesian statistics** is an approach to inferential statistics based on Bayes' theorem, where available knowledge about
# parameters in a statistical model is updated with the information in observed data. The background knowledge is expressed
# as a prior distribution and combined with observational data in the form of a likelihood function to determine the posterior
# distribution. The posterior can also be used for making predictions about future events.

# **Bayesian statistics** is a departure from classical inferential statistics that prohibits probability statements about
# parameters and is based on asymptotically sampling infinite samples from a theoretical population and finding parameter values
# that maximize the likelihood function. Mostly notorious is null-hypothesis significance testing (NHST) based on *p*-values.
# Bayesian statistics **incorporate uncertainty** (and prior knowledge) by allowing probability statements about parameters,
# and the process of parameter value inference is a direct result of the **Bayes' theorem**.

# Bayesian statistics is **revolutionizing all fields of evidence-based science**[^evidencebased] (van de Schoot et al. 2021).
# Dissatisfaction with traditional methods of statistical inference (frequentist statistics) and the advent of computers with exponential
# growth in computing power[^computingpower] provided a rise in Bayesian statistics because it is an approach aligned with the human
# intuition of uncertainty, robust to scientific malpractices, but computationally intensive.

# But before we get into Bayesian statistics, we have to talk about **probability**: the engine of Bayesian inference.

# ## What is Probability?

# > PROBABILITY DOES NOT EXIST!
# >
# > de Finetti (1974)[^deFinetti]

# These are the first words in the preface to the famous book by [Bruno de Finetti](https://en.wikipedia.org/wiki/Bruno_de_Finetti)
# (figure below), one of the most important probability mathematician and philosopher.
# Yes, probability does not exist. Or rather, probability as a physical quantity, objective chance, **does NOT exist**.
# De Finetti showed that, in a precise sense, if we dispense with the question of objective chance *nothing is lost*.
# The mathematics of inductive reasoning remains **exactly the same**.

# ![De Finetti](/pages/images/finetti.jpg)

# Consider tossing a weighted coin. The attempts are considered independent and, as a result, exhibit another
# important property: **the order does not matter**. To say that order does not matter is to say that if you take any
# finite sequence of heads and tails and exchange the results however you want,
# the resulting sequence will have the same probability. We say that this probability is
# **invariant under permutations**.

# Or, to put it another way, the only thing that matters is the relative frequency.
# Result that have the same frequency of heads and tails consequently have the same probability.
# The frequency is considered a **sufficient statistic**. Saying that order doesn't matter or saying
# that the only thing that matters is frequency are two ways of saying exactly the same thing.
# This property is called **exchangeability** by de Finetti. And it is the most important property of
# probability that makes it possible for us to manipulate it mathematically (or philosophically) even
# if it does not exist as a physical "thing".

# Still developing the argument:
# > "Probabilistic reasoning - always understood as subjective - stems
# > merely stems from our being uncertain about something. It makes no difference whether the uncertainty
# > relates to an unforeseeable future [^subjective], or to an unnoticed past, or to a past doubtfully reported
# > or forgotten [^objective]... The only relevant thing is uncertainty - the extent of our own knowledge and ignorance.
# > The actual fact of whether or not the events considered are in some sense determined, or known by other people,
# > and so on, is of no consequence." (de Finetti, 1974)

# In conclusion: no matter what the probability is, you can use it anyway, even if it is an absolute frequency
# (ex: probability that I will ride by bike naked is ZERO because the probability that an event that never occurred
# will occur in the future it is ZERO) or a subjective guess (ex: maybe the probability is not ZERO, but 0.00000000000001;
# very unlikely, but not impossible).

# ### Mathematical Definition

# ## Footnotes
#
# [^evidencebased]: personally, like a good Popperian, I don't believe there is science without being evidence-based; what does not use evidence can be considered as logic, philosophy or social practices (no less or more important than science, just a demarcation of what is science and what is not; eg, mathematics and law).
# [^computingpower]: your smartphone (iPhone 12 - 4GB RAM) has 1,000,000x (1 million) more computing power than the computer that was aboard the Apollo 11 (4kB RAM) which took the man to the moon. Detail: this on-board computer was responsible for lunar module navigation, route and controls.
# [^deFinetti]: if the reader wants an in-depth discussion see Nau (2001).
# [^subjective]: my observation: related to the subjective Bayesian approach.
# [^objective]: my observation: related to the objective frequentist approach.

# ## References
#
# Amrhein, V., Greenland, S., & McShane, B. (2019). Scientists rise up against statistical significance. *Nature*, 567(7748), 305–307. https://doi.org/10.1038/d41586-019-00857-9
#
# Baird, D. (1983). The fisher/pearson chi-squared controversy: A turning point for inductive inference. *The British Journal for the Philosophy of Science*, 34(2), 105–118.
#
# Benjamin, D. J., Berger, J. O., Johannesson, M., Nosek, B. A., Wagenmakers, E.-J., Berk, R., … Johnson, V. E. (2018). Redefine statistical significance. *Nature Human Behaviour*, 2(1), 6–10. https://doi.org/10.1038/s41562-017-0189-z
#
# de Finetti, B. (1974). *Theory of Probability*. New York: John Wiley & Sons.
#
# Eckhardt, R. (1987). Stan Ulam, John von Neumann, and the Monte Carlo Method. *Los Alamos Science*, 15(30), 131–136.
#
# Fisher, R. A. (1925). *Statistical methods for research workers*. Oliver; Boyd.
#
# Fisher, R. A. (1962). Some Examples of Bayes’ Method of the Experimental Determination of Probabilities A Priori. *Journal of the Royal Statistical Society. Series B (Methodological)*, 24(1), 118–124. Retrieved from https://www.jstor.org/stable/2983751
#
# Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis*. Chapman and Hall/CRC.
#
# Goodman, S. N. (2016). Aligning statistical and scientific reasoning. *Science*, 352(6290), 1180–1181. https://doi.org/10.1126/science.aaf5406
#
# Head, M. L., Holman, L., Lanfear, R., Kahn, A. T., & Jennions, M. D. (2015). The extent and consequences of p-hacking in science. *PLoS Biol*, 13(3), e1002106.
#
# Ioannidis, J. P. A. (2019). What Have We (Not) Learnt from Millions of Scientific Papers with <i>P</i> Values? *The American Statistician*, 73(sup1), 20–25. https://doi.org/10.1080/00031305.2018.1447512
#
# It’s time to talk about ditching statistical significance. (2019). *Nature*, 567(7748, 7748), 283–283. https://doi.org/10.1038/d41586-019-00874-8
#
# Jaynes, E. T. (2003). *Probability theory: The logic of science*. Cambridge university press.
#
# Kolmogorov, A. N. (1933). *Foundations of the Theory of Probability*. Berlin: Julius Springer.
#
# Lakens, D., Adolfi, F. G., Albers, C. J., Anvari, F., Apps, M. A. J., Argamon, S. E., … Zwaan, R. A. (2018). Justify your alpha. *Nature Human Behaviour*, 2(3), 168–171. https://doi.org/10.1038/s41562-018-0311-x
#
# Nau, R. F. (2001). De Finetti was Right: Probability Does Not Exist. *Theory and Decision*, 51(2), 89–124. https://doi.org/10.1023/A:1015525808214
#
# Neyman, J. (1937). Outline of a theory of statistical estimation based on the classical theory of probability. *Philosophical Transactions of the Royal Society of London*. Series A, Mathematical and Physical Sciences, 236(767), 333–380.
#
# Neyman, J., & Pearson, E. S. (1933). On the problem of the most efficient tests of statistical hypotheses. *Philosophical Transactions of the Royal Society of London*. Series A, Containing Papers of a Mathematical or Physical Character, 231(694-706), 289–337.
#
# Rosnow, R. L., & Rosenthal, R. (1989). Statistical procedures and the justification of knowledge in psychological science. *American Psychologist*, 44, 1276–1284.
#
# Stigler, S. M., & others. (2007). The epic story of maximum likelihood. *Statistical Science*, 22(4), 598–620.
#
# van de Schoot, R., Depaoli, S., King, R., Kramer, B., Märtens, K., Tadesse, M. G., … Yau, C. (2021). Bayesian statistics and modelling. *Nature Reviews Methods Primers*, 1(1, 1), 1–26. https://doi.org/10.1038/s43586-020-00001-2
#
# Wasserstein, R. L., & Lazar, N. A. (2016). The ASA’s Statement on p-Values: Context, Process, and Purpose. *American Statistician*, 70(2), 129–133. https://doi.org/10.1080/00031305.2016.1154108
#
# Wasserstein, R. L., Schirm, A. L., & Lazar, N. A. (2019). Moving to a World Beyond "p < 0.05." *American Statistician*, 73, 1–19. https://doi.org/10.1080/00031305.2019.1583913
