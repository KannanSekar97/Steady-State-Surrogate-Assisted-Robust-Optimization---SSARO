import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def nondominated(F):
    """
  
    Parameters
    ----------
    F : ndarray -> Objective values [n_solutions, n_obj]
    
    Returns
    -------
    ndarray
        nondominated set.
    """

    nds = NonDominatedSorting().do(F, n_stop_if_ranked=2)
    front1_nds = nds[0]
    if len(front1_nds) == 1 and len(nds) > 1:
        combined_nds = np.concatenate([front1_nds, nds[1]])
        return F[combined_nds], combined_nds
    else:
        return F[front1_nds], front1_nds
