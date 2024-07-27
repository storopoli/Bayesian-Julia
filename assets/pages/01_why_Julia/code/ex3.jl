# This file was generated, do not modify it. # hide
import Base: size, getindex

struct OneHotVector <: AbstractVector{Int}
    len::Int
    ind::Int
end

size(v::OneHotVector) = (v.len,)

getindex(v::OneHotVector, i::Integer) = Int(i == v.ind)