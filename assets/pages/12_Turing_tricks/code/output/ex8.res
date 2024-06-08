Chains MCMC chain (1000×20×4 Array{Float64, 3}):

Iterations        = 1001:1:2000
Number of chains  = 4
Samples per chain = 1000
parameters        = real_β[1], real_β[2], real_β[3], α, β[1], β[2], β[3], σ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters       mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol    Float64   Float64   Float64     Float64     Float64   Float64       Missing

   real_β[1]     6.2103    2.1904    0.0339   4190.5377   2075.4888    1.0015       missing
   real_β[2]     0.5062    0.0616    0.0015   1657.4420   2135.3528    1.0021       missing
   real_β[3]    -0.0701    0.2143    0.0058   1370.8734   1800.6496    1.0010       missing
           α    32.9057    7.8274    0.2470   1006.3265   1242.3293    1.0026       missing
        β[1]   -49.9922    7.0245    0.2210   1009.4898   1401.5358    1.0022       missing
        β[2]    22.0858    3.5735    0.1101   1054.2580   1418.8568    1.0028       missing
        β[3]     0.2869    0.8775    0.0238   1370.8734   1800.6496    1.0010       missing
           σ    17.8703    0.5859    0.0113   2699.9100   2464.9195    1.0019       missing

Quantiles
  parameters       2.5%      25.0%      50.0%      75.0%      97.5%
      Symbol    Float64    Float64    Float64    Float64    Float64

   real_β[1]     1.8485     4.7441     6.2339     7.7050    10.3766
   real_β[2]     0.3815     0.4656     0.5071     0.5480     0.6252
   real_β[3]    -0.5252    -0.2045    -0.0659     0.0728     0.3310
           α    17.6798    27.6922    32.7076    38.1740    47.9793
        β[1]   -63.6746   -54.6970   -50.1683   -45.2700   -36.2948
        β[2]    15.1066    19.7019    22.1804    24.4485    29.1569
        β[3]    -1.3554    -0.2981     0.2697     0.8374     2.1505
           σ    16.7293    17.4690    17.8683    18.2608    19.0084
