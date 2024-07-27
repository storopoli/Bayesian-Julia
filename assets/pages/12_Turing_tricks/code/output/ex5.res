Chains MCMC chain (1000×17×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 5.73 seconds
Compute duration  = 21.56 seconds
parameters        = α, β[1], β[2], β[3], σ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64

           α   21.3876    8.4858    0.2149   1559.9041   2213.4495    1.0022       72.3686
        β[1]    2.0034    1.8156    0.0436   2236.2241   1738.3535    1.0005      103.7450
        β[2]    0.5813    0.0592    0.0014   1804.9360   2249.5101    1.0017       83.7363
        β[3]    0.2506    0.3068    0.0071   1873.4478   2288.8003    1.0039       86.9148
           σ   17.8739    0.5871    0.0118   2494.7722   2480.9154    1.0004      115.7398

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           α    4.7959   15.6521   21.4281   27.2003   37.4842
        β[1]   -0.5505    0.6853    1.6233    2.9332    6.3939
        β[2]    0.4655    0.5419    0.5807    0.6197    0.6977
        β[3]   -0.3429    0.0395    0.2445    0.4523    0.8689
           σ   16.7445   17.4648   17.8620   18.2550   19.0900
