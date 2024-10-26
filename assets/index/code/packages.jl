# This file was generated, do not modify it. # hide
# hideall
import Pkg

deps = [pair.second for pair in Pkg.dependencies()]
deps = filter(p -> p.is_direct_dep, deps)
deps = filter(p -> !isnothing(p.version), deps)
list = ["$(p.name) $(p.version)" for p in deps]
sort!(list)
println(join(list, '\n'))