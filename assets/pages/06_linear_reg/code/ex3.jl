# This file was generated, do not modify it. # hide
X = Matrix(select(kidiq, Not(:kid_score)))
y = kidiq[:, :kid_score]
model = linreg(X, y);