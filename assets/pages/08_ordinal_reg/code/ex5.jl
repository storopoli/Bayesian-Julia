# This file was generated, do not modify it. # hide
using CategoricalArrays

DataFrames.transform!(
    esoph,
    :agegp =>
        x -> categorical(
            x; levels=["25-34", "35-44", "45-54", "55-64", "65-74", "75+"], ordered=true
        ),
    :alcgp =>
        x -> categorical(x; levels=["0-39g/day", "40-79", "80-119", "120+"], ordered=true),
    :tobgp =>
        x -> categorical(x; levels=["0-9g/day", "10-19", "20-29", "30+"], ordered=true);
    renamecols=false,
)
DataFrames.transform!(
    esoph, [:agegp, :alcgp, :tobgp] .=> ByRow(levelcode); renamecols=false
)

X = Matrix(select(esoph, [:agegp, :alcgp]))
y = esoph[:, :tobgp]
model = ordreg(X, y);