# This file was generated, do not modify it.

abstract type Pet end
struct Dog <: Pet
    name::String
end
struct Cat <: Pet
    name::String
end

function encounter(a::Pet, b::Pet)
    verb = meets(a, b)
    return println("$(a.name) meets $(b.name) and $verb")
end

meets(a::Dog, b::Dog) = "sniffs";
meets(a::Dog, b::Cat) = "chases";
meets(a::Cat, b::Dog) = "hisses";
meets(a::Cat, b::Cat) = "slinks";

fido = Dog("Fido");
rex = Dog("Rex");
whiskers = Cat("Whiskers");
spots = Cat("Spots");

encounter(fido, rex)
encounter(rex, whiskers)
encounter(spots, fido)
encounter(whiskers, spots)

import Base: size, getindex

struct OneHotVector <: AbstractVector{Int}
    len::Int
    ind::Int
end

size(v::OneHotVector) = (v.len,)

getindex(v::OneHotVector, i::Integer) = Int(i == v.ind)

onehot = [OneHotVector(3, rand(1:3)) for _ in 1:4]

using LinearAlgebra

function inner_sum(A, vs)
    t = zero(eltype(A))
    for v in vs
        t += inner(v, A, v) # multiple dispatch!
    end
    return t
end

inner(v, A, w) = dot(v, A * w) # very general definition

A = rand(3, 3)
vs = [rand(3) for _ in 1:4]
inner_sum(A, vs)

supertype(OneHotVector)

inner_sum(A, onehot)

using BenchmarkTools

@btime inner_sum($A, $onehot);

import Base: *

*(A::AbstractMatrix, v::OneHotVector) = A[:, v.ind]
inner(v::OneHotVector, A, w::OneHotVector) = A[v.ind, w.ind]

@btime inner_sum($A, $onehot);
