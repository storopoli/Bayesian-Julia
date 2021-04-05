using Pkg, Literate

Pkg.activate(".")

scripts = joinpath.("_literate", readdir("_literate"))

nbpath = joinpath("pages")

for file in scripts
   # Generate annotated notebooks
    Literate.markdown(file, nbpath)
end
