# This file was generated, do not modify it. # hide
X_met_1 = metropolis(
    S_parallel, width, ρ; seed=124, start_x=first(starts[1]), start_y=last(starts[1])
);
X_met_2 = metropolis(
    S_parallel, width, ρ; seed=125, start_x=first(starts[2]), start_y=last(starts[2])
);
X_met_3 = metropolis(
    S_parallel, width, ρ; seed=126, start_x=first(starts[3]), start_y=last(starts[3])
);
X_met_4 = metropolis(
    S_parallel, width, ρ; seed=127, start_x=first(starts[4]), start_y=last(starts[4])
);