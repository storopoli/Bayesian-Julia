@def title = "Bayesian Statistics using Julia and Turing"
@def tags = ["syntax", "code"]

# Bayesian Statistics using Julia and Turing

![Bayesian for Everyone](images/bayes-meme.jpg)

\tableofcontents <!-- you can use \toc as well -->

Welcome to the repository of tutorials on how to Bayesian Statistics using [Julia](https://www.julialang.org) and [Turing](http://turing.ml/). Tutorials are available at [storopoli.io/Bayesian-Julia](https://storopoli.io/Bayesian-Julia).

Bayesian statistics is an approach to inferential statistics based on Bayes' theorem, where available knowledge about parameters in a statistical model is updated with the information in observed data. The background knowledge is expressed as a prior distribution and combined with observational data in the form of a likelihood function to determine the posterior distribution. The posterior can also be used for making predictions about future events.

Bayesian statistics is a departure from classical inferential statistics that prohibits probability statements about parameters and is based on asymptotically sampling infinite samples from a theoretical population and finding parameter values that maximize the likelihood function. Mostly notorious is null-hypothesis significance testing (NHST) based on *p*-values. Bayesian statistics incorporate uncertainty (and prior knowledge) by allowing probability statements about parameters, and the process of parameter value inference is a direct result of the Bayes' theorem.

## Turing

[Turing](http://turing.ml/) is a ecosystem of Julia packages for Bayesian Inference using [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming). Models specified using Turing are easy to read and write — models work the way you write them. Like everything in Julia, Turing is [fast](https://arxiv.org/abs/2002.02702).

## Author

José Eduardo Storopoli, PhD - [*Lattes* CV](http://lattes.cnpq.br/2281909649311607) - [ORCID](https://orcid.org/0000-0002-0559-5176) - <https://storopoli.io>

## How to use the content?

The content is licensed under a very permissive Creative Commons license (CC BY-SA). You are mostly welcome to contribute with [issues](https://www.github.com/storopoli/Bayesian-Julia/issues) and [pull requests](https://github.com/storopoli/Bayesian-Julia/pulls). My hope is to have more people into Bayesian statistics. The content is aimed towards social scientists and PhD candidates in social sciences. I chose to provide an intuitive approach rather than focusing on rigorous mathematical formulations. I've made it to be how I would have liked to be introduced to Bayesian statistics.

To configure a local environment:

1. Download and install [Julia](https://www.julialang.org/downloads/)
2.  Clone the repository from GitHub:
    `git clone https://github.com/storopoli/Bayesian-Julia.git`
3.  Access the directory: `cd Bayesian-Julia`
4.  Activate the environment: type inside the Julia REPL `]activate .`

## Tutorials

1. [**Why Julia?**](placeholder)
2. [**What is Bayesian Statistics**](placeholder)
3. [**Common Probability Distributions**](placeholder)
4. [**How to use Turing**](placeholder)
5. [**Markov Chain Monte Carlo (MCMC)**](placeholder)
6. [**Bayesian Linear Regression**](placeholder)
7. [**Bayesian Logistic Regression**](placeholder)
8. [**Bayesian Regression with Count Data**](placeholder)
9. [**Robust Bayesian Regression**](placeholder)
10. [**Multilevel Models (a.k.a. Hierarchical Models)**](placeholder)
11. [**Computational Tricks with Turing (Non-Centered Parametrization and QR Decomposition)**](placeholder)

## What about other Turing tutorials?

Despite not being the only Turing tutorial that exists, this tutorial aims to introduce Bayesian inference along with how to use Julia and Turing. Here is a (not complete) list of other Turing tutorials:

1. [**Official Turing Tutorials**](https://turing.ml/dev/tutorials/): tutorials on how to implement common models in Turing
2. [**Statistical Rethinking - Turing Models**](https://statisticalrethinkingjulia.github.io/TuringModels.jl/): Julia versions of the Bayesian models described in *Statistical Rethinking* Edition 1 (McElreath, 2016) and Edition 2 (McElreath, 2020)
3. [**Håkan Kjellerstrand Turing Tutorials**](http://hakank.org/julia/turing/): a collection of Julia Turing models

## How to cite

To cite these tutorials, please use:

```plaintext
Storopoli (2021). Bayesian Statistics with Julia and Turing. https://storopoli.io/Bayesian-Julia.
```

Or in BibTeX format $\LaTeX$:

```plaintext
@misc{storopoli2021bayesianjulia,
      author = {Storopoli, Jose},
      title = {Bayesian Statistics with Julia and Turing},
      url = {https://storopoli.io/Bayesian-Julia},
      year = {2021}
    }
```

## References

### Books

-   Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis*. Chapman and
    Hall/CRC.
-   McElreath, R. (2020). *Statistical rethinking: A Bayesian course with examples in R and Stan*. CRC press.
-   Gelman, A., Hill, J., & Vehtari, A. (2020). *Regression and other stories*. Cambridge University Press.
-   Brooks, S., Gelman, A., Jones, G., & Meng, X.-L. (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press. <https://books.google.com?id=qfRsAIKZ4rIC>
    -   Geyer, C. J. (2011). Introduction to markov chain monte carlo. In S. Brooks, A. Gelman, G. L. Jones, & X.-L. Meng (Eds.), *Handbook of markov chain monte carlo*.

### Academic Papers

-   van de Schoot, R., Depaoli, S., King, R., Kramer, B., Märtens, K., Tadesse, M. G., Vannucci, M., Gelman, A., Veen, D., Willemsen, J., & Yau, C. (2021). Bayesian statistics and modelling. *Nature Reviews Methods Primers*, *1*(1, 1), 1–26. <https://doi.org/10.1038/s43586-020-00001-2>
-   Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in Bayesian workflow. *Journal of the Royal Statistical Society: Series A (Statistics in Society)*, *182*(2), 389–402. <https://doi.org/10.1111/rssa.12378>
-   Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modr’ak, M. (2020, November 3). *Bayesian Workflow*. <http://arxiv.org/abs/2011.01808>
-   Benjamin, D. J., Berger, J. O., Johannesson, M., Nosek, B. A., Wagenmakers, E.-J., Berk, R., Bollen, K. A., Brembs, B., Brown, L., Camerer, C., Cesarini, D., Chambers, C. D., Clyde, M., Cook, T. D., De Boeck, P., Dienes, Z., Dreber, A., Easwaran, K., Efferson, C., … Johnson, V. E. (2018). Redefine statistical significance. *Nature Human Behaviour*, *2*(1), 6–10. <https://doi.org/10.1038/s41562-017-0189-z>
-   McShane, B. B., Gal, D., Gelman, A., Robert, C., & Tackett, J. L. (2019). Abandon Statistical Significance. *American Statistician*, *73*, 235–245. <https://doi.org/10.1080/00031305.2018.1527253>
-   Amrhein, V., Greenland, S., & McShane, B. (2019). Scientists rise up against statistical significance. *Nature*, *567*(7748), 305–307. <https://doi.org/10.1038/d41586-019-00857-9>
-   van de Schoot, R., Kaplan, D., Denissen, J., Asendorpf, J. B., Neyer, F. J., & van Aken, M. A. G. (2014). A Gentle Introduction to Bayesian Analysis: Applications to Developmental Research. *Child Development*, *85*(3), 842–860. <https://doi.org/10.1111/cdev.12169>

## License

This content is licensed under [Creative Commons Attribution-ShareAlike 4.0 Internacional](http://creativecommons.org/licenses/by-sa/4.0/).
