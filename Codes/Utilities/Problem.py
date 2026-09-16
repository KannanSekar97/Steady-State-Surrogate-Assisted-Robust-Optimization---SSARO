import numpy as np
from pymoo.core.problem import Problem
from joblib import Parallel, delayed 

# ==========================================================
# Select ~95% dominated sample
# ==========================================================
def domination_count(F):
    """
    Exact dominator count for 2-objective minimization.
    O(N log N), much faster than pairwise O(N^2). Fenwick approach
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]

    f2_vals, f2_rank = np.unique(F[:, 1], return_inverse=True)
    tree = np.zeros(len(f2_vals) + 1, dtype=int)

    def add(i, v=1):
        i += 1
        while i < len(tree):
            tree[i] += v
            i += i & -i

    def sum_leq(i):
        i += 1
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    counts = np.zeros(n, dtype=int)
    order = np.lexsort((F[:, 1], F[:, 0]))

    start = 0
    while start < n:
        end = start
        f1 = F[order[start], 0]

        while end < n and F[order[end], 0] == f1:
            end += 1

        group = order[start:end]
        group_sorted = group[np.argsort(F[group, 1])]

        # prior groups: f1 smaller, f2 <= current
        for idx in group_sorted:
            counts[idx] += sum_leq(f2_rank[idx])

        # same f1 group: only smaller f2 dominates
        _, first_pos = np.unique(F[group_sorted, 1], return_index=True)
        for pos, idx in enumerate(group_sorted):
            counts[idx] += first_pos[np.searchsorted(F[group_sorted[first_pos], 1], F[idx, 1])]

        for idx in group:
            add(f2_rank[idx])

        start = end

    return counts

# ==========================================================
# Select ~95% dominated sample
# ==========================================================
def select_alpha_dominated_feasible_or_least_infeasible(
    F,
    G,
    alpha=0.95,
    seed=None
):
    """
    Select an alpha-percentile dominated objective sample.

    Tie-breaking:
        If multiple solutions are equally close to the alpha-percentile
        domination count, one is selected uniformly at random.

    Handles:
        1. Unconstrained problems
        2. Constrained problems with feasible samples
        3. Constrained problems with no feasible samples
    """

    rng = np.random.default_rng(seed)

    F = np.asarray(F, dtype=float)
    G = np.asarray(G, dtype=float)

    def select_by_alpha_random_tie(F_sub, counts, alpha):
        counts = np.asarray(counts, dtype=float)

        q = np.quantile(counts, alpha)
        dist = np.abs(counts - q)

        candidate_idx = np.where(dist == dist.min())[0]

        return rng.choice(candidate_idx)

    # ==================================================
    # Unconstrained problem
    # ==================================================
    if G.size == 0:
        if alpha == "mean":
            return np.mean(F, axis=0)

        counts = domination_count(F)

        local_idx = select_by_alpha_random_tie(F, counts, alpha)
        return F[local_idx]

    # ==================================================
    # Constrained problem
    # ==================================================
    if G.ndim == 1:
        G = G.reshape(-1, 1)

    feasible_mask = np.all(G <= 0.0, axis=1)

    # ==================================================
    # If feasible samples exist
    # ==================================================
    if np.any(feasible_mask):
        feasible_indices = np.where(feasible_mask)[0]
        F_feas = F[feasible_mask]
        M = F_feas.shape[0]

        if alpha == "mean":
            return np.mean(F_feas, axis=0)

        if M == 1:
            return F_feas[0]

        counts = domination_count(F_feas)

        local_idx = select_by_alpha_random_tie(F_feas, counts, alpha)
        return F[feasible_indices[local_idx]]

    # ==================================================
    # If no feasible sample exists
    # choose least-infeasible sample
    # ==================================================
    CV = np.sum(np.maximum(G, 0.0), axis=1)

    min_cv = np.min(CV)
    candidate_idx = np.where(CV == min_cv)[0]

    least_infeasible_idx = rng.choice(candidate_idx)

    return F[least_infeasible_idx]

def process_design(F_samples_i, G_samples_i, alpha):

    if alpha == "mean":
        return np.mean(F_samples_i, axis=0)

    return select_alpha_dominated_feasible_or_least_infeasible(
        F_samples_i,
        G_samples_i,
        alpha=alpha,
    )

def sample_random_variables(random_variables, shape):
    """
    shape = (N, nsamples)
    returns array of shape (N, nsamples, n_random)
    """

    if len(random_variables) == 0:
        return None

    samples = []

    for rv in random_variables:
        dist = rv["distribution"].lower()
        if dist == "normal":
            values = np.random.normal(
                rv["mean"],
                rv["std"],
                shape
            )

        elif dist == "uniform":
            values = np.random.uniform(
                rv["lower"],
                rv["upper"],
                shape
            )

        else:
            raise ValueError(
                f"Unsupported distribution: {rv['distribution']}"
            )

        samples.append(values)

    return np.stack(samples, axis=2)


# ==========================================================
# Problem
# ==========================================================
class MyProblem(Problem):

    def __init__(
        self,
        evaluator,
        concept,
        del_x=0.01,
        alpha=0.95,
        nsamples_scalar=100,
        reliability_target=0.98,
    ):

        self.concept = concept
        self.evaluator = evaluator

        self.n_design = concept["nvar"]
        self.n_obj = concept["nf"]
        self.n_con = concept["nc"]

        self.random_variables = concept.get("random_variables", [])
        self.n_random = len(self.random_variables)

        self.has_random = self.n_random > 0

        bounds = np.asarray(concept["bounds"], dtype=float)

        super().__init__(
            n_var=self.n_design,
            n_obj=self.n_obj,
            n_ieq_constr=self.n_con,
            xl=bounds[:, 0],
            xu=bounds[:, 1],
        )

        self.del_x = del_x
        self.alpha = alpha
        self.nsamples = nsamples_scalar*(self.n_design + self.n_random)
        self.k_dom = nsamples_scalar*self.n_design
        self.reliability_target = reliability_target
        self.bound = bounds
        # print(nsamples_scalar)
        # print(f"number of perturbed samples: {self.nsamples}")
    def _evaluate(self, x, out, *args, **kwargs):

        x = np.atleast_2d(x[:, :self.n_design])
        N = x.shape[0]
        ns = self.nsamples

        # ==================================================
        # Case 1: deterministic problem, no robustness
        # ==================================================
        if self.del_x == 0 and not self.has_random:

            out["F"] = self.evaluator.eval_obj(x)

            if self.n_con > 0:
                out["G"] = self.evaluator.eval_con(x)
            else:
                out["G"] = np.empty((N, 0))

            return

        # ==================================================
        # Design perturbation samples
        # ==================================================
        if self.del_x == 0:
            design_samples = np.repeat(x[:, None, :], ns, axis=1)

        else:
            xl = self.bound[:, 0]
            xu = self.bound[:,1]
            delta = self.del_x * (xu - xl)
            # print(delta)
            lower = np.maximum(x[:, None, :] - delta[None, None, :], xl[None, None, :])
            upper = np.minimum(x[:, None, :] + delta[None, None, :], xu[None, None, :])

            u = np.random.rand(N, ns, self.n_design)
            design_samples = lower + u * (upper - lower)

        # ==================================================
        # Random variables
        # ==================================================
        if self.has_random:
            random_samples = sample_random_variables(self.random_variables, shape=(N, ns))
            X_all = np.concatenate([design_samples, random_samples], axis=2)

        else:
            X_all = design_samples

        # ==================================================
        # Surrogate input
        # ==================================================
        X_flat = X_all.reshape( N * ns, self.n_design + self.n_random)
        # ==================================================
        # Objective evaluation
        # ==================================================
        F_samples = self.evaluator.eval_obj(X_flat)
        F_samples = np.asarray(F_samples).reshape(N, ns, self.n_obj)

        # ==================================================
        # Constraint evaluation
        # ==================================================
        if self.n_con > 0:
            G_samples = self.evaluator.eval_con(X_flat)
            G_samples = np.asarray(G_samples).reshape(N, ns, self.n_con)

        else:
            G_samples = np.empty((N, ns, 0))



        # ==================================================
        # Feasibility-aware robust objective selection
        # ==================================================
        robust_F = Parallel(n_jobs=-1,
            backend="loky",
        )(
            delayed(process_design)(F_samples[i], G_samples[i], self.alpha)for i in range(N))
        out["F"] = np.asarray(robust_F)

        # ==================================================
        # Constraint output
        # ==================================================
        if self.n_con == 0:
            out["G"] = np.empty((N, 0))

        else:
            reliability = np.mean(G_samples <= 0.0,axis=1)
            out["G"] = self.reliability_target - reliability