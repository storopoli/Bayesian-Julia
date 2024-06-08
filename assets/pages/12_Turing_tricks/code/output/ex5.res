Chains MCMC chain (1000×17×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 5.92 seconds
Compute duration  = 22.64 seconds
parameters        = α, β[1], β[2], β[3], σ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64

           α   21.5126    8.6720    0.2217   1528.8886   1868.2612    1.0009       67.5304
        β[1]    2.0014    1.7943    0.0419   2281.1940   1734.6512    1.0016      100.7595
        β[2]    0.5788    0.0584    0.0013   2163.9754   2292.8814    1.0006       95.5820
        β[3]    0.2566    0.3092    0.0074   1762.0214   2135.6795    1.0010       77.8278
           σ   17.8859    0.6033    0.0106   3271.1669   2347.2435    1.0008      144.4862

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           α    4.7278   15.7633   21.2942   27.4322   38.4426
        β[1]   -0.5876    0.7324    1.6761    2.9919    6.3388
        β[2]    0.4662    0.5392    0.5793    0.6184    0.6924
        β[3]   -0.3477    0.0440    0.2588    0.4733    0.8490
           σ   16.7525   17.4685   17.8796   18.2703   19.1238
