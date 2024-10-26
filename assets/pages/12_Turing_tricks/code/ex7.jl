# This file was generated, do not modify it. # hide
model_qr = linreg(Q_ast, y)
chain_qr = sample(model_qr, NUTS(1_000, 0.65), MCMCThreads(), 1_000, 4)