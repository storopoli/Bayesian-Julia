# This file was generated, do not modify it. # hide
Q, R = qr(X)
Q_ast = Matrix(Q) * sqrt(size(X, 1) - 1)
R_ast = R / sqrt(size(X, 1) - 1);