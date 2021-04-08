# # # Why Julia?

# [Julia](https://www.julialang.org) is a relatively new language, first released in 2012, aims to be both high-level and fast.
# Julia is a fast dynamic-typed language that just-in-time (JIT)
# compiles into native code using LLVM. It ["runs like C but reads like Python"](https://www.nature.com/articles/d41586-019-02310-3),
# meaning that is *blazing* fast, easy to prototype and read/write code.
# It is multi-paradigm, combining features of imperative, functional, and object-oriented programming.

# From my point-of-view julia has three main features that makes it a unique language to work with, specially in scientific computing:

# * Speed
# * Ease of Use
# * Multiple Dispatch

# ## Speed

# Yes, Julia as fast. Very fast! It was made for speed from the drawing board. It bypass any sort of intermediate representation and translate
# code into machine native code using LLVM compiler. Comparing this with R, that uses either FORTRAN or C, or Python, that uses CPython;
# and you'll clearly see that Julia has a major speed advantage over other languages that are common in data science and statistics.
# Julia exposes the machine code to LLVM's compiler which in turn can optimize code as it wishes, like a good compiler such as LLVM excels in.

# For example, NASA uses Julia to Analyze the
# "[Largest Batch of Earth-Sized Planets Ever Found](https://exoplanets.nasa.gov/news/1669/seven-rocky-trappist-1-planets-may-be-made-of-similar-stuff/)"
# The analysis is conducted using Julia.

# ## Ease of Use

# What is most striking that Julia can be as fast as C (and faster than Java in some applications) while having a very simple and
# intelligible syntax. This feature along with its speed is what Julia creators denote as "the two language problem" that Julia
# address. The "two language problem" is a very typical situation in scientific computing where a researcher or computer scientist
# devises an algorithm or a solution that he or she prototypes in an easy to code language (like Python) and, if it works, he or she
# would code in a fast language that is not easy to code (C or FORTRAN). Thus, we have two languages involved in the process of
# of developing a new solution. One which is easy to prototype but is not suited for implementation (mostly due to  being slow).
# And another one which is not so easy to code (and, consequently, not easy to prototype) but suited for implementation
# (mostly because it is fast). Julia comes to eliminate such situations by being the same language that you prototype (ease of use)
# and implement the solution (speed).

# Also, Julia lets you use unicode characters as variables or parameters. This means no more `sigma` or `sigma_i`,
# and instead just use `σ` or `σᵢ` as you would in mathematical notation. When you see code for an algorithm or for a
# mathematical equation you see a one-to-one relation to code and math. This is a powerful feature.

# I think that the "two language problem" and the one-to-one code and math relation are best described by
# one of the creators of Julia, Alan Edelman, in a TED Talk (see the video below):

# ~~~
# <iframe width="560" height="315" src="https://www.youtube.com/embed/qGW0GT1rCvs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
# ~~~

# ## Multiple Dispatch

# ~~~
# <iframe width="560" height="315" src="https://www.youtube.com/embed/kc9HwsxE1OY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
# ~~~
