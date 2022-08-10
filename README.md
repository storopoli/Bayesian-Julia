# Bayesian Statistics using Julia and Turing

[![CC BY-SA
4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)

<div class="figure" style="text-align: center">

<img src="images/bayes-meme.jpg" alt="Bayesian for Everyone!" width="500" />
<p class="caption">
Bayesian for Everyone!
</p>

</div>

Welcome to the repository of tutorials on how to do **Bayesian Statistics** using [**Julia**](https://www.julialang.org) and [**Turing**](http://turing.ml/).
Tutorials are available at [storopoli.github.io/Bayesian-Julia](https://storopoli.github.io/Bayesian-Julia).

**Bayesian statistics** is an approach to inferential statistics based on Bayes' theorem,
where available knowledge about parameters in a statistical model is updated with the information in observed data.
The background knowledge is expressed as a prior distribution and combined with observational data in the form of a likelihood function to determine the posterior distribution.
The posterior can also be used for making predictions about future events.

**Bayesian statistics** is a departure from classical inferential statistics that prohibits probability statements about parameters and
is based on asymptotically sampling infinite samples from a theoretical population and finding parameter values that maximize the likelihood function.
Mostly notorious is null-hypothesis significance testing (NHST) based on *p*-values.
Bayesian statistics **incorporate uncertainty** (and prior knowledge) by allowing probability statements about parameters,
and the process of parameter value inference is a direct result of the **Bayes' theorem**.

## Table of Contents

   * [Julia](#julia)
   * [Turing](#turing)
   * [Author](#author)
   * [How to use the content?](#how-to-use-the-content)
   * [Tutorials](#tutorials)
   * [What about other Turing tutorials?](#what-about-other-turing-tutorials)
   * [How to cite](#how-to-cite)
   * [References](#references)
      * [Books](#books)
      * [Academic Papers](#academic-papers)
         * [Primary](#primary)
         * [Auxiliary](#auxiliary)
   * [License](#license)

## Julia

[**Julia**](https://www.julialang.org) is a fast dynamic-typed language that just-in-time (JIT) compiles into native code using LLVM.
It ["runs like C but reads like Python"](https://www.nature.com/articles/d41586-019-02310-3),
meaning that is *blazing* fast, easy to prototype and to read/write code.
It is multi-paradigm, combining features of imperative, functional, and object-oriented programming.
I won't cover Julia basics and any sort of data manipulation using Julia in the tutorials,
instead please take a look into the following resources which covers most of the introduction to Julia and how to work with tabular data in Julia:

* [**Julia Documentation**](https://docs.julialang.org/): Julia documentation is a very friendly and well-written resource that explains the basic design and functionality of the language.
* [**Julia Data Science**](https://juliadatascience.io): open source and open access book on how to do Data Science using Julia.
* [**Thinking Julia**](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html): introductory beginner-friendly book that explains the main concepts and functionality behind the Julia language.
* [**Julia High Performance**](https://www.amazon.com/Julia-High-Performance-Avik-Sengupta/dp/178829811X): book by two of the creators of the Julia Language ([Avik Sengupta](https://www.linkedin.com/in/aviks) and [Alan Edelman](http://www-math.mit.edu/~edelman/)), it covers how to make Julia even faster with some principles and tricks of the trade.
* [**An Introduction DataFrames**](https://github.com/bkamins/Julia-DataFrames-Tutorial): the package [`DataFrames.jl`](https://dataframes.juliadata.org/stable/) provides a set of tools for working with tabular data in Julia. Its design and functionality are similar to those of `pandas` (in Python) and `data.frame`, `data.table` and `dplyr` (in R), making it a great general purpose data science tool, especially for those coming to Julia from R or Python.This is a collection of notebooks that introduces `DataFrames.jl` made by one of its core contributors [Bogumił Kamiński](https://github.com/bkamins).

## Turing

[**Turing**](http://turing.ml/) is an ecosystem of Julia packages for Bayesian Inference using [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming).
Models specified using Turing are easy to read and write — models work the way you write them. Like everything in Julia, Turing is [fast](https://arxiv.org/abs/2002.02702).

## Author

Jose Storopoli, PhD - [*Lattes* CV](http://lattes.cnpq.br/2281909649311607) - [ORCID](https://orcid.org/0000-0002-0559-5176) - <https://storopoli.io>

## How to use the content?

The content is licensed under a very permissive Creative Commons license (CC BY-SA).
You are mostly welcome to contribute with [issues](https://www.github.com/storopoli/Bayesian-Julia/issues) and [pull requests](https://github.com/storopoli/Bayesian-Julia/pulls).
My hope is to have **more people into Bayesian statistics**. The content is aimed towards social scientists and PhD candidates in social sciences.
I chose to provide an **intuitive approach** rather than focusing on rigorous mathematical formulations.
I've made it to be how I would have liked to be introduced to Bayesian statistics.

To configure a local environment:

1. Download and install [Julia](https://www.julialang.org/downloads/)
2.  Clone the repository from GitHub:
    `git clone https://github.com/storopoli/Bayesian-Julia.git`
3.  Access the directory: `cd Bayesian-Julia`
4.  Activate the environment by typing in the Julia REPL:
    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

## Tutorials

1. [**Why Julia?**](https://storopoli.github.io/Bayesian-Julia/pages/01_why_Julia/)
2. [**What is Bayesian Statistics?**](https://storopoli.github.io/Bayesian-Julia/pages/02_bayes_stats/)
3. [**Common Probability Distributions**](https://storopoli.github.io/Bayesian-Julia/pages/03_prob_dist/)
4. [**How to use Turing**](https://storopoli.github.io/Bayesian-Julia/pages/04_Turing/)
5. [**Markov Chain Monte Carlo (MCMC)**](https://storopoli.github.io/Bayesian-Julia/pages/05_MCMC/)
6. [**Bayesian Linear Regression**](https://storopoli.github.io/Bayesian-Julia/pages/06_linear_reg/)
7. [**Bayesian Logistic Regression**](https://storopoli.github.io/Bayesian-Julia/pages/07_logistic_reg/)
8. [**Bayesian Ordinal Regression**](https://storopoli.github.io/Bayesian-Julia/pages/08_ordinal_reg/)
9. [**Bayesian Regression with Count Data**](https://storopoli.github.io/Bayesian-Julia/pages/09_count_reg/)
10. [**Robust Bayesian Regression**](https://storopoli.github.io/Bayesian-Julia/pages/10_robust_reg/)
11. [**Multilevel Models (a.k.a. Hierarchical Models)**](https://storopoli.github.io/Bayesian-Julia/pages/11_multilevel_models/)
12. [**Computational Tricks with Turing (Non-Centered Parametrization and QR Decomposition)**](https://storopoli.github.io/Bayesian-Julia/pages/12_Turing_tricks/)
13. [**Epidemiological Models using ODE Solvers in Turing**](https://storopoli.github.io/Bayesian-Julia/pages/13_epi_models/)

## Datasets

- `kidiq` (linear regression): data from a survey of adult American women and their children
   (a subsample from the National Longitudinal Survey of Youth).
   Source: Gelman and Hill (2007).
- `wells` (logistic regression): a survey of 3200 residents in a small area of Bangladesh suffering
   from arsenic contamination of groundwater.
   Respondents with elevated arsenic levels in their wells had been encouraged to switch their water source
   to a safe public or private well in the nearby area
   and the survey was conducted several years later to
   learn which of the affected residents had switched wells.
   Souce: Gelman and Hill (2007).
- `esoph` (ordinal regression): data from a case-control study of (o)esophageal cancer in Ille-et-Vilaine, France.
   Source: Breslow and Day (1980).
- `roaches` (Poisson regression): data on the efficacy of a pest management system at reducing the number of roaches in urban apartments.
   Source: Gelman and Hill (2007).
- `duncan` (robust regression): data from occupation's prestige filled with outliers.
   Source: Duncan (1961).
- `cheese` (hierarchical models): data from cheese ratings.
   A group of 10 rural and 10 urban raters rated 4 types of different cheeses (A, B, C and D) in two samples.
   Source: Boatwright, McCulloch and Rossi (1999).

## What about other Turing tutorials?

Despite not being the only Turing tutorial that exists, this tutorial aims to introduce Bayesian inference along with how to use Julia and Turing.
Here is a (not complete) list of other Turing tutorials:

1. [**Official Turing Tutorials**](https://turing.ml/dev/tutorials/): tutorials on how to implement common models in Turing
2. [**Statistical Rethinking - Turing Models**](https://statisticalrethinkingjulia.github.io/TuringModels.jl/): Julia versions of the Bayesian models described in *Statistical Rethinking* Edition 1 (McElreath, 2016) and Edition 2 (McElreath, 2020)
3. [**Håkan Kjellerstrand Turing Tutorials**](http://hakank.org/julia/turing/): a collection of Julia Turing models

I also have a free and opensource graduate course on Bayesian Statistics with Turing and Stan code.
You can find it at [`storopoli/Bayesian-Statistics`](https://github.com/storopoli/Bayesian-Statistics).

## How to cite

To cite these tutorials, please use:

    Storopoli (2021). Bayesian Statistics with Julia and Turing. https://storopoli.github.io/Bayesian-Julia.

Or in BibTeX format (LaTeX):

    @misc{storopoli2021bayesianjulia,
      author = {Storopoli, Jose},
      title = {Bayesian Statistics with Julia and Turing},
      url = {https://storopoli.github.io/Bayesian-Julia},
      year = {2021}
    }

## References

The references are divided in **books**, **papers**, **software**, and **datasets**.

### Books

* Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis*. Chapman and Hall/CRC.
* McElreath, R. (2020). *Statistical rethinking: A Bayesian course with examples in R and Stan*. CRC press.
* Gelman, A., Hill, J., & Vehtari, A. (2020). *Regression and other stories*. Cambridge University Press.
* Brooks, S., Gelman, A., Jones, G., & Meng, X.-L. (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press. <https://books.google.com?id=qfRsAIKZ4rIC>
    * Geyer, C. J. (2011). Introduction to markov chain monte carlo. In S. Brooks, A. Gelman, G. L. Jones, & X.-L. Meng (Eds.), *Handbook of markov chain monte carlo*.

### Academic Papers

* van de Schoot, R., Depaoli, S., King, R., Kramer, B., Märtens, K., Tadesse, M. G., Vannucci, M., Gelman, A., Veen, D., Willemsen, J., & Yau, C. (2021). Bayesian statistics and modelling. *Nature Reviews Methods Primers*, *1*(1, 1), 1–26. <https://doi.org/10.1038/s43586-020-00001-2>
* Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in Bayesian workflow. *Journal of the Royal Statistical Society: Series A (Statistics in Society)*, *182*(2), 389–402. <https://doi.org/10.1111/rssa.12378>
* Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modr’ak, M. (2020, November 3). *Bayesian Workflow*. <http://arxiv.org/abs/2011.01808>
* Benjamin, D. J., Berger, J. O., Johannesson, M., Nosek, B. A., Wagenmakers, E.-J., Berk, R., Bollen, K. A., Brembs, B., Brown, L., Camerer, C., Cesarini, D., Chambers, C. D., Clyde, M., Cook, T. D., De Boeck, P., Dienes, Z., Dreber, A., Easwaran, K., Efferson, C., … Johnson, V. E. (2018). Redefine statistical significance. *Nature Human Behaviour*, *2*(1), 6–10. <https://doi.org/10.1038/s41562-017-0189-z>
* McShane, B. B., Gal, D., Gelman, A., Robert, C., & Tackett, J. L. (2019). Abandon Statistical Significance. *American Statistician*, *73*, 235–245. <https://doi.org/10.1080/00031305.2018.1527253>
* Amrhein, V., Greenland, S., & McShane, B. (2019). Scientists rise up against statistical significance. *Nature*, *567*(7748), 305–307. <https://doi.org/10.1038/d41586-019-00857-9>
* van de Schoot, R., Kaplan, D., Denissen, J., Asendorpf, J. B., Neyer, F. J., & van Aken, M. A. G. (2014). A Gentle Introduction to Bayesian Analysis: Applications to Developmental Research. *Child Development*, *85*(3), 842–860. <https://doi.org/10.1111/cdev.12169>

### Software

* Bezanson, J., Edelman, A., Karpinski, S., & Shah, V. B. (2017). Julia: A fresh approach to numerical computing. SIAM Review, 59(1), 65–98.
* Ge, H., Xu, K., & Ghahramani, Z. (2018). Turing: A Language for Flexible Probabilistic Inference. International Conference on Artificial Intelligence and Statistics, 1682–1690. http://proceedings.mlr.press/v84/ge18b.html
* Tarek, M., Xu, K., Trapp, M., Ge, H., & Ghahramani, Z. (2020). DynamicPPL: Stan-like Speed for Dynamic Probabilistic Models. ArXiv:2002.02702 [Cs, Stat]. http://arxiv.org/abs/2002.02702
* Xu, K., Ge, H., Tebbutt, W., Tarek, M., Trapp, M., & Ghahramani, Z. (2020). AdvancedHMC.jl: A robust, modular and efficient implementation of advanced HMC algorithms. Symposium on Advances in Approximate Bayesian Inference, 1–10. http://proceedings.mlr.press/v118/xu20a.html
* Revels, J., Lubin, M., & Papamarkou, T. (2016). Forward-Mode Automatic Differentiation in Julia. ArXiv:1607.07892 [Cs]. http://arxiv.org/abs/1607.07892

### Datasets

-   Boatwright, P., McCulloch, R., & Rossi, P. (1999). Account-level modeling for trade promotion: An application of a constrained parameter hierarchical model. _Journal of the American Statistical Association_, 94(448), 1063–1073.
-   Breslow, N. E. & Day, N. E. (1980). **Statistical Methods in Cancer Research. Volume 1: The Analysis of Case-Control Studies**. IARC Lyon / Oxford University Press.
-   Duncan, O. D. (1961). A socioeconomic index for all occupations. Class: Critical Concepts, 1, 388–426.
-   Gelman, A., & Hill, J. (2007). **Data analysis using regression and
    multilevel/hierarchical models**. Cambridge university press.

## License

This content is licensed under [Creative Commons Attribution-ShareAlike 4.0 Internacional](http://creativecommons.org/licenses/by-sa/4.0/).

[![CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
