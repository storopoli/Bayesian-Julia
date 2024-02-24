# This file was generated, do not modify it. # hide
let
    probs = [0.10, 0.15, 0.33, 0.25, 0.10, 0.07]
    dist = Categorical(probs)
    x = 1:length(probs)
    x_pmf = pdf.(dist, x)
    x_cdf = cdf.(dist, x)
    x_logodds_cdf = logit.(x_cdf)
    df = DataFrame(; x, x_pmf, x_cdf, x_logodds_cdf)
    labels = ["CDF", "Log-cumulative-odds"]
    f = Figure()
    plt1 = data(df) * mapping(:x, :x_pmf) * visual(BarPlot)
    plt2 =
        data(df) *
        mapping(:x, [:x_cdf, :x_logodds_cdf]; col=dims(1) => renamer(labels)) *
        visual(ScatterLines)
    axis = (; xticks=1:6)
    draw!(f[1, 2:3], plt1; axis)
    draw!(f[2, 1:4], plt2; axis, facet=(; linkyaxes=:none))
    f
    save(joinpath(@OUTPUT, "logodds.svg"), f) # hide
end