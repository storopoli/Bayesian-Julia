Chains MCMC chain (1000×21×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 11.72 seconds
Compute duration  = 43.58 seconds
parameters        = α, β[1], β[2], β[3], β[4], σ, τ, zⱼ[1], zⱼ[2]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters       mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol    Float64   Float64   Float64     Float64     Float64   Float64       Float64

           α    70.7400    3.9162    0.1535    746.8037    618.6365    1.0057       17.1356
        β[1]     2.9733    1.3479    0.0298   2059.4494   2126.2529    1.0029       47.2546
        β[2]   -10.5540    1.3858    0.0332   1731.0925   1336.6279    1.0007       39.7204
        β[3]     6.5563    1.3427    0.0309   1890.8060   2006.9763    1.0038       43.3850
        β[4]     1.0767    1.3266    0.0329   1631.6887   1056.7471    1.0036       37.4395
           σ     7.4178    0.4592    0.0097   2204.0103   1429.4954    1.0007       50.5716
           τ     5.3524    3.0071    0.1366    762.4344    292.4834    1.0041       17.4942
       zⱼ[1]    -0.8283    0.7924    0.0211   1343.0493   1518.7369    1.0018       30.8166
       zⱼ[2]     0.8513    0.7943    0.0221   1225.0626    684.1272    1.0049       28.1094

Quantiles
  parameters       2.5%      25.0%      50.0%     75.0%     97.5%
      Symbol    Float64    Float64    Float64   Float64   Float64

           α    62.4757    68.4153    70.7528   72.9883   78.9826
        β[1]     0.3948     2.0540     2.9718    3.8870    5.6591
        β[2]   -13.2620   -11.4840   -10.5581   -9.6211   -7.8268
        β[3]     3.8493     5.6694     6.5733    7.4445    9.1352
        β[4]    -1.5665     0.2066     1.0831    1.9586    3.6893
           σ     6.5692     7.0980     7.4000    7.7030    8.3854
           τ     1.8545     3.1768     4.5002    6.6171   13.7044
       zⱼ[1]    -2.4668    -1.3391    -0.8119   -0.2700    0.6011
       zⱼ[2]    -0.6738     0.2976     0.8311    1.3566    2.4837
