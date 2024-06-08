Chains MCMC chain (1000×21×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 7.07 seconds
Compute duration  = 24.45 seconds
parameters        = α, β[1], β[2], β[3], β[4], σ, τ, αⱼ[1], αⱼ[2]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters       mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol    Float64   Float64   Float64     Float64     Float64   Float64       Float64

           α    70.6388    5.4554    0.2162    915.6164    759.2940    1.0028       37.4470
        β[1]     2.9639    1.3435    0.0265   2565.3867   2773.4856    1.0012      104.9195
        β[2]   -10.5915    1.3904    0.0277   2522.8976   2504.4545    1.0016      103.1818
        β[3]     6.5485    1.3619    0.0264   2668.6532   2413.9865    0.9997      109.1429
        β[4]     1.0905    1.3750    0.0285   2338.0486   2471.5094    1.0010       95.6218
           σ     7.4087    0.4507    0.0088   2643.0429   2581.2624    1.0001      108.0955
           τ     6.2641    5.7017    0.1952   1222.7006   1203.5556    1.0038       50.0062
       αⱼ[1]    -3.3228    5.3516    0.2167    865.3478    780.3551    1.0027       35.3911
       αⱼ[2]     3.7386    5.3657    0.2173    894.2647    746.3891    1.0034       36.5737

Quantiles
  parameters       2.5%      25.0%      50.0%     75.0%     97.5%
      Symbol    Float64    Float64    Float64   Float64   Float64

           α    59.8857    68.3215    70.7533   73.2141   81.3633
        β[1]     0.3781     2.0754     2.9757    3.8536    5.5841
        β[2]   -13.2884   -11.5392   -10.6060   -9.6587   -7.8122
        β[3]     3.9443     5.6304     6.5351    7.4435    9.2425
        β[4]    -1.6999     0.1515     1.1072    2.0425    3.7349
           σ     6.5891     7.0826     7.3872    7.7026    8.3437
           τ     1.8181     3.2859     4.6436    7.1806   20.4141
       αⱼ[1]   -14.0825    -5.6883    -3.3988   -1.1424    7.5331
       αⱼ[2]    -6.5136     1.3252     3.5179    5.9215   14.7231
