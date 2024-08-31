# This file was generated, do not modify it. # hide
X = Matrix(select(roaches, Not(:y)))
y = roaches[:, :y]
model_poisson = poissonreg(X, y);