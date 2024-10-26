Chains MCMC chain (1000×17×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 5.79 seconds
Compute duration  = 21.28 seconds
parameters        = α, β[1], β[2], β[3], σ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64

           α   21.4873    8.4924    0.2037   1745.2565   2027.6065    1.0016       82.0062
        β[1]    2.0094    1.8046    0.0404   2503.6573   1943.0201    1.0018      117.6420
        β[2]    0.5804    0.0587    0.0013   2105.4795   2393.3903    1.0002       98.9324
        β[3]    0.2501    0.3040    0.0064   2289.2311   2357.9811    1.0011      107.5665
           σ   17.8704    0.6117    0.0128   2302.7508   2294.8862    1.0010      108.2018

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           α    5.1041   15.6692   21.4971   27.2446   38.0485
        β[1]   -0.6083    0.7132    1.6417    3.0215    6.2792
        β[2]    0.4648    0.5429    0.5800    0.6190    0.6949
        β[3]   -0.3675    0.0436    0.2546    0.4551    0.8387
           σ   16.7198   17.4475   17.8650   18.2730   19.1362
