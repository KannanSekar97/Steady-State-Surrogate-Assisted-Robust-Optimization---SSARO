
# Robust reference ND front generation
import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2 
# Problems
from Problems.Unconstrained.test_problem_4 import TEST_PROBLEM_4
from Problems.Unconstrained.rtp1 import RTP1
from Problems.Unconstrained.rtp2 import RTP2
from Problems.Constrained.cbeam import CANTILEVER_BEAM
from Problems.Constrained.torque_arm import TORQUE_ARM
from Problems.Constrained.bwb_vsp_aero_problem import BWB_VSP_AERO

#Utilities
from Utilities.Problem_Robust_True import MyProblem_Robust
# from Utilities.Problem_Robust_True_BWB import MyProblem_Robust   # Onyl for BWB_VSP_AERO
concepts = [RTP1]

del_x = 0.01  #+/- 1% of range
alpha = 0.95
nsamples_scalar = 100
for concept in concepts:
    nvar, nf, nc, bounds = concept["nvar"], concept["nf"], concept["nc"], concept["bounds"]
    lb, ub = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    problem = MyProblem_Robust(concept, del_x, alpha, nsamples_scalar)

    algorithm = NSGA2(pop_size=100)
    res_true = minimize(problem,
                        algorithm,
                        ('n_gen', 100),
                        seed=1,
                        verbose=True
                        )
    True_Pareto = res_true.F

    plt.scatter(True_Pareto[:, 0], True_Pareto[:, 1], label=f'{concept["name"]}')
plt.legend()
plt.show()
np.save(f"Robust_front_{concept['name']}_X.npy", res_true.X)
np.save(f"Robust_front_{concept['name']}_F.npy", res_true.F)
np.save(f"Robust_front_{concept['name']}_G.npy", res_true.G)