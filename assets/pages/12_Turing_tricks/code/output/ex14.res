Chains MCMC chain (1000×21×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 6.74 seconds
Compute duration  = 25.04 seconds
parameters        = α, β[1], β[2], β[3], β[4], σ, τ, αⱼ[1], αⱼ[2]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters       mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol    Float64   Float64   Float64     Float64     Float64   Float64       Float64

           α    70.9777    4.9292    0.1951    733.8674    698.0036    1.0024       29.3043
        β[1]     2.9391    1.3125    0.0274   2286.1309   2670.5875    1.0023       91.2882
        β[2]   -10.6031    1.3439    0.0273   2408.5309   2595.0412    1.0021       96.1758
        β[3]     6.5407    1.2973    0.0254   2620.3949   2658.5828    1.0009      104.6358
        β[4]     1.0906    1.3048    0.0260   2514.3608   2958.6202    1.0011      100.4017
           σ     7.3923    0.4537    0.0079   3320.4428   2540.7254    1.0014      132.5897
           τ     6.5988    7.6336    0.2042   1628.0444   1702.8711    1.0011       65.0100
       αⱼ[1]    -3.6532    4.8490    0.1914    742.0644    627.3595    1.0028       29.6316
       αⱼ[2]     3.4087    4.8273    0.1883    759.3680    689.2538    1.0033       30.3226

Quantiles
  parameters       2.5%      25.0%      50.0%     75.0%     97.5%
      Symbol    Float64    Float64    Float64   Float64   Float64

           α    60.9306    68.3731    70.8225   73.3482   81.9235
        β[1]     0.3449     2.0539     2.9806    3.8499    5.4024
        β[2]   -13.2616   -11.5028   -10.6004   -9.7101   -7.9419
        β[3]     4.0352     5.6609     6.5454    7.4219    9.0464
        β[4]    -1.4362     0.2293     1.0859    1.9292    3.6770
           σ     6.5463     7.0668     7.3869    7.6893    8.3178
           τ     1.8032     3.2176     4.6419    7.4369   21.6780
       αⱼ[1]   -14.3009    -5.8219    -3.4560   -1.2685    6.2597
       αⱼ[2]    -7.2993     1.1619     3.4004    5.7226   13.5513
