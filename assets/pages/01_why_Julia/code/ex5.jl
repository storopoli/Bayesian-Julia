# This file was generated, do not modify it. # hide
using LinearAlgebra

function inner_sum(A, vs)
    t = zero(eltype(A))
    for v in vs
        t += inner(v, A, v) # multiple dispatch!
    end
    return t
end

inner(v, A, w) = dot(v, A * w) # very general definition