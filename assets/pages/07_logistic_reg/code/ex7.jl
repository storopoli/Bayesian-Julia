# This file was generated, do not modify it. # hide
function logodds2prob(logodds::Float64)
    return exp(logodds) / (1 + exp(logodds))
end

@chain quantile(chain) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(logodds2prob); renamecols=false)
end