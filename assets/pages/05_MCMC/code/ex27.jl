# This file was generated, do not modify it. # hide
X_gibbs_1 = gibbs(
    S_parallel * 2, ρ; seed=124, start_x=first(starts[1]), start_y=last(starts[1])
);
X_gibbs_2 = gibbs(
    S_parallel * 2, ρ; seed=125, start_x=first(starts[2]), start_y=last(starts[2])
);
X_gibbs_3 = gibbs(
    S_parallel * 2, ρ; seed=126, start_x=first(starts[3]), start_y=last(starts[3])
);
X_gibbs_4 = gibbs(
    S_parallel * 2, ρ; seed=127, start_x=first(starts[4]), start_y=last(starts[4])
);