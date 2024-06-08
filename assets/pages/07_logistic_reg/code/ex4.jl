# This file was generated, do not modify it. # hide
X = Matrix(select(wells, Not(:switch)))
y = wells[:, :switch]
model = logreg(X, y);