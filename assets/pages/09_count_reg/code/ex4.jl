# This file was generated, do not modify it. # hide
function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    p = p > 0 ? p : 1e-4 # numerical stability
    r = ϕ

    return NegativeBinomial(r, p)
end