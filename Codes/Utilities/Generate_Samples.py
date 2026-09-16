import numpy as np
from smt.sampling_methods import LHS

def generate_samples(seed, xlimits, n_sample):
    xlimits = np.array(xlimits)

    # First try: random_state (works for your SMT)
    try:
        rng = np.random.RandomState(seed)
        lhs = LHS(
            xlimits=xlimits,
            criterion="maximin",
            random_state=rng
        )
    except Exception:
        # Fallback (future SMT versions)
        lhs = LHS(
            xlimits=xlimits,
            criterion="maximin"
        )
        np.random.seed(seed)

    return lhs(n_sample)


# import numpy as np
# from scipy.stats import qmc

# def generate_samples(seed, xlimits, n_sample):
#     xlimits = np.array(xlimits)
#     dim = xlimits.shape[0]

#     sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    
#     sample = sampler.random(n=n_sample)
    
#     # Scale to your bounds
#     sample_scaled = qmc.scale(sample, xlimits[:, 0], xlimits[:, 1])
    
#     return sample_scaled