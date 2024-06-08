Chains MCMC chain (1000×21×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 10.66 seconds
Compute duration  = 41.11 seconds
parameters        = α, β[1], β[2], β[3], β[4], σ, τ, zⱼ[1], zⱼ[2]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters       mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol    Float64   Float64   Float64     Float64     Float64   Float64       Float64

           α    71.1250    3.5397    0.1218    878.2014    784.3499    1.0054       21.3633
        β[1]     2.9093    1.2858    0.0297   1876.0233   2481.1601    1.0026       45.6365
        β[2]   -10.6841    1.3789    0.0608    571.3700    192.4652    1.0074       13.8992
        β[3]     6.5318    1.3379    0.0326   1713.4983   1455.9135    1.0030       41.6828
        β[4]     1.0746    1.3279    0.0316   1767.8948   2009.5112    1.0071       43.0061
           σ     7.3765    0.4561    0.0158    777.6440    188.6146    1.0139       18.9171
           τ     5.0157    2.6654    0.1354    604.8463    193.9385    1.0075       14.7136
       zⱼ[1]    -0.9156    0.7846    0.0225   1184.6917   1334.6682    1.0041       28.8190
       zⱼ[2]     0.8326    0.8046    0.0223   1253.2053    947.4690    1.0023       30.4857

Quantiles
  parameters       2.5%      25.0%      50.0%     75.0%     97.5%
      Symbol    Float64    Float64    Float64   Float64   Float64

           α    63.6608    68.9974    71.0956   73.1433   78.5094
        β[1]     0.3584     2.0549     2.9066    3.7865    5.4293
        β[2]   -13.5656   -11.5643   -10.6594   -9.7752   -7.9711
        β[3]     3.9994     5.5582     6.4857    7.4393    9.0957
        β[4]    -1.4433     0.2116     1.0321    1.9658    3.6415
           σ     6.5332     7.0626     7.3592    7.6827    8.3100
           τ     1.8047     3.1023     4.2758    6.3272   11.7515
       zⱼ[1]    -2.5756    -1.4160    -0.8862   -0.3486    0.5239
       zⱼ[2]    -0.6131     0.2661     0.7795    1.3822    2.4805
