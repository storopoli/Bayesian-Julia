# This file was generated, do not modify it. # hide
@chain quantile(chain_negbin) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(exp); renamecols=false)
end