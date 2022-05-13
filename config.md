<!--
Add here global page variables to use throughout your
website.
The website_* must be defined for the RSS to work
-->
@def website_title = "Bayesian Statistics with Julia and Turing"
@def website_descr = "Bayesian Statistics Tutorials for Social Scientists and PhD Candidates"
@def website_url   = "https://storopoli.github.io/Bayesian-Julia/"

@def author = "Jose Storopoli"

@def mintoclevel = 2

@def prepath = "Bayesian-Julia"

<!--
Add here files or directories that should be ignored by Franklin, otherwise
these files might be copied and, if markdown, processed by Franklin which
you might not want. Indicate directories by ending the name with a `/`.
-->
@def ignore = ["node_modules/", "franklin", "franklin.pub", "datasets/"]

<!--
Add here global latex commands to use throughout your
pages. It can be math commands but does not need to be.
For instance:
* \newcommand{\phrase}{This is a long phrase to copy.}
-->
\newcommand{\R}{\mathbb R}
\newcommand{\scal}[1]{\langle #1 \rangle}
\newcommand{\note}[1]{@@note @@title ⚠ Note@@ @@content #1 @@ @@}
\newcommand{\warn}[1]{@@warning @@title ⚠ Warning!@@ @@content #1 @@ @@}
\newcommand{\center}[1]{@@text-center #1 @@}
