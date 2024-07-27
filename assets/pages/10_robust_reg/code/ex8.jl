# This file was generated, do not modify it. # hide
X = Matrix(select(duncan, [:income, :education]))
y = duncan[:, :prestige]
model = robustreg(X, y);