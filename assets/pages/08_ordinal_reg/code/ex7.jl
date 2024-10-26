# This file was generated, do not modify it. # hide
using Chain

@chain quantile(chain) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(exp); renamecols=false)
end