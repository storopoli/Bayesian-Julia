Chains MCMC chain (1000×20×4 Array{Float64, 3}):

Iterations        = 1001:1:2000
Number of chains  = 4
Samples per chain = 1000
parameters        = real_β[1], real_β[2], real_β[3], α, β[1], β[2], β[3], σ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters       mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol    Float64   Float64   Float64     Float64     Float64   Float64       Missing

   real_β[1]     6.2260    2.2090    0.0338   4290.3240   2898.0017    1.0011       missing
   real_β[2]     0.5025    0.0621    0.0015   1766.4415   2072.6612    1.0014       missing
   real_β[3]    -0.0775    0.2193    0.0056   1576.3622   1631.1408    1.0010       missing
           α    33.4170    7.7313    0.2345   1075.0809   1406.4488    1.0028       missing
        β[1]   -49.5148    6.9240    0.2101   1071.5267   1340.7849    1.0027       missing
        β[2]    21.8468    3.5239    0.1059   1098.9101   1451.6713    1.0029       missing
        β[3]     0.3173    0.8981    0.0230   1576.3622   1631.1408    1.0010       missing
           σ    17.8535    0.5970    0.0126   2267.5644   2154.7581    1.0026       missing

Quantiles
  parameters       2.5%      25.0%      50.0%      75.0%      97.5%
      Symbol    Float64    Float64    Float64    Float64    Float64

   real_β[1]     1.8824     4.7474     6.2627     7.7091    10.5882
   real_β[2]     0.3770     0.4631     0.5057     0.5439     0.6236
   real_β[3]    -0.5431    -0.2089    -0.0645     0.0699     0.3272
           α    18.8769    28.1126    33.1124    38.4653    49.3102
        β[1]   -62.3996   -54.2390   -49.8054   -44.8671   -35.2573
        β[2]    14.5922    19.5482    21.9376    24.2182    28.3814
        β[3]    -1.3400    -0.2862     0.2641     0.8554     2.2238
           σ    16.7007    17.4506    17.8256    18.2409    19.1256
