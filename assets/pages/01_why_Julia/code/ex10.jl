# This file was generated, do not modify it. # hide
import Base: *

*(A::AbstractMatrix, v::OneHotVector) = A[:, v.ind]
inner(v::OneHotVector, A, w::OneHotVector) = A[v.ind, w.ind]