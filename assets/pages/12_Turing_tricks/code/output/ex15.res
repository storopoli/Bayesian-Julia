Chains MCMC chain (1000×21×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 11.28 seconds
Compute duration  = 41.56 seconds
parameters        = α, β[1], β[2], β[3], β[4], σ, τ, zⱼ[1], zⱼ[2]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters       mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol    Float64   Float64   Float64     Float64     Float64   Float64       Float64

           α    70.9900    4.8647    0.4635    155.4578     88.8729    1.0302        3.7402
        β[1]     2.9527    1.2926    0.0330   1500.3760   2639.7845    1.0026       36.0980
        β[2]   -10.6579    1.3714    0.0349   1626.5273   1511.2828    1.0048       39.1331
        β[3]     6.5143    1.3092    0.0386   1145.7270   1584.8775    1.0084       27.5654
        β[4]     1.0844    1.3234    0.0336   1538.1462   2459.6317    1.0037       37.0067
           σ     7.3943    0.4490    0.0162    694.4465    349.2168    1.0055       16.7079
           τ     5.6634    3.1771    0.2281    291.3404    130.2148    1.0067        7.0094
       zⱼ[1]    -0.8149    0.8581    0.0567    290.1429     89.2838    1.0145        6.9806
       zⱼ[2]     0.7927    0.8278    0.0519    250.4265    177.2815    1.0201        6.0251

Quantiles
  parameters       2.5%      25.0%      50.0%     75.0%     97.5%
      Symbol    Float64    Float64    Float64   Float64   Float64

           α    57.9767    68.6853    71.0670   73.6389   80.7890
        β[1]     0.3888     2.1078     3.0012    3.8348    5.3833
        β[2]   -13.1715   -11.5936   -10.7263   -9.7402   -7.8199
        β[3]     3.9367     5.6344     6.5416    7.3767    9.0813
        β[4]    -1.5205     0.1702     1.0809    1.9515    3.7650
           σ     6.5671     7.0758     7.3812    7.6904    8.3126
           τ     1.8948     3.3209     4.7284    7.2258   14.3458
       zⱼ[1]    -2.4355    -1.3699    -0.8245   -0.2901    1.2010
       zⱼ[2]    -0.6772     0.1950     0.7770    1.3361    2.4660
