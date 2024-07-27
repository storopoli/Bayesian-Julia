Chains MCMC chain (1000×21×4 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 4
Samples per chain = 1000
Wall duration     = 7.47 seconds
Compute duration  = 26.92 seconds
parameters        = α, β[1], β[2], β[3], β[4], σ, τ, αⱼ[1], αⱼ[2]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters       mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec
      Symbol    Float64   Float64   Float64     Float64     Float64   Float64       Float64

           α    70.7913    6.3145    0.3002    638.6275    549.4570    1.0012       23.7196
        β[1]     2.9636    1.3457    0.0279   2332.9421   2647.4652    1.0009       86.6492
        β[2]   -10.5873    1.3666    0.0278   2416.3746   2770.1755    1.0018       89.7480
        β[3]     6.5473    1.3280    0.0255   2716.3239   2707.3158    1.0013      100.8886
        β[4]     1.0912    1.3143    0.0269   2380.6035   2572.9234    1.0000       88.4194
           σ     7.3953    0.4545    0.0082   3130.7519   2452.7010    1.0012      116.2811
           τ     6.8382    7.5177    0.2971    991.7078    724.7443    1.0006       36.8336
       αⱼ[1]    -3.4850    6.2621    0.2972    656.6183    526.4821    1.0019       24.3878
       αⱼ[2]     3.5999    6.2423    0.2975    653.5929    549.2077    1.0019       24.2755

Quantiles
  parameters       2.5%      25.0%      50.0%     75.0%     97.5%
      Symbol    Float64    Float64    Float64   Float64   Float64

           α    59.1725    68.2517    70.7931   73.3145   82.5962
        β[1]     0.2909     2.0538     2.9881    3.9033    5.4819
        β[2]   -13.2229   -11.5118   -10.6103   -9.6965   -7.8916
        β[3]     3.9740     5.6547     6.5489    7.4807    9.0881
        β[4]    -1.5472     0.2266     1.1099    1.9647    3.6794
           σ     6.5576     7.0837     7.3674    7.6830    8.3523
           τ     1.8109     3.2661     4.7466    7.6220   24.5143
       αⱼ[1]   -15.5049    -5.8453    -3.3990   -1.1592    8.4981
       αⱼ[2]    -8.0026     1.2141     3.4957    5.9211   15.7215
