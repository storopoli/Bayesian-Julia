Chains MCMC chain (1000×20×4 Array{Float64, 3}):

Iterations        = 1001:1:2000
Number of chains  = 4
Samples per chain = 1000
parameters        = real_β[1], real_β[2], real_β[3], α, β[1], β[2], β[3], σ
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters       mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol    Float64   Float64   Float64     Float64     Float64   Float64       Missing

   real_β[1]     6.2873    2.1921    0.0337   4243.2988   2940.8659    1.0004       missing
   real_β[2]     0.5022    0.0651    0.0020   1157.6335   1467.5847    1.0015       missing
   real_β[3]    -0.0821    0.2299    0.0074   1099.7874   1210.5649    1.0017       missing
           α    33.5261    8.5202    0.3294    762.1540    758.6985    1.0060       missing
        β[1]   -49.4478    7.6440    0.2944    770.0258    774.9488    1.0070       missing
        β[2]    21.7894    3.8757    0.1479    785.6259    729.7449    1.0060       missing
        β[3]     0.3363    0.9415    0.0301   1099.7874   1210.5649    1.0017       missing
           σ    17.8645    0.5924    0.0122   2337.3604   2264.5760    1.0027       missing

Quantiles
  parameters       2.5%      25.0%      50.0%      75.0%      97.5%
      Symbol    Float64    Float64    Float64    Float64    Float64

   real_β[1]     2.0723     4.7898     6.2768     7.7708    10.5720
   real_β[2]     0.3696     0.4602     0.5035     0.5471     0.6245
   real_β[3]    -0.5577    -0.2136    -0.0686     0.0661     0.3302
           α    18.4458    27.8121    33.1619    38.7305    50.9329
        β[1]   -63.1518   -54.6239   -49.7727   -44.8569   -33.7062
        β[2]    13.6960    19.4252    21.9738    24.4045    28.7370
        β[3]    -1.3522    -0.2708     0.2808     0.8746     2.2836
           σ    16.7411    17.4558    17.8498    18.2630    19.0793
