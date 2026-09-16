import numpy as np

class ConceptEvaluator:

    def __init__(self, concept, nf, nc,
                 PySR_obj, PySR_con):
        
        self.concept = concept
        self.nf = nf
        self.nc = nc
        self.PySR_obj = PySR_obj
        self.PySR_con = PySR_con

    # ===============================================================
    # Evaluate Objectives f1, f2, ... using TRUE or Surrogate (SA)
    # ===============================================================
    def eval_obj(self, x):
        x = np.atleast_2d(x)
        outputs = []
        for i in range(self.nf):
            pred = self.PySR_obj[i].predict(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            outputs.append(np.ravel(pred))
        return np.column_stack(outputs)

    # ===============================================================
    # Evaluate Constraints g1, g2, ... using TRUE or Surrogate (SA)
    # ===============================================================
    def eval_con(self, x):
        x = np.atleast_2d(x)
        outputs = []
        if self.nc == 0:
            return np.empty((x.shape[0], 0))
        for j in range(self.nc):
            pred = self.PySR_con[j].predict(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            outputs.append(np.ravel(pred))
        return np.column_stack(outputs)
    
     # ===============================================================
    # Evaluate Objectives f1, f2, ... using TRUE or Surrogate (SA)
    # ===============================================================
    # def eval_obj(self, x):
    #     obj_modes = ["SA"] * self.nf  # or "SA" for surrogate
    #     x = np.atleast_2d(x)
    #     outputs = []
    #     for i, mode in enumerate(obj_modes):

    #         if mode == "TRUE":
    #             # call individual f[i]
    #             fi_value = self.concept["f"][i](x)
    #             outputs.append(np.ravel(fi_value))

    #         elif mode == "SA":
    #             pred = self.PySR_obj[i].predict(x)
    #             if isinstance(pred, tuple):
    #                 pred = pred[0]
    #             outputs.append(np.ravel(pred))

    #         else:
    #             raise ValueError(f"Unknown mode {mode} in obj_modes")

    #     return np.column_stack(outputs)

    # # ===============================================================
    # # Evaluate Constraints g1, g2, ... using TRUE or Surrogate (SA)
    # # ===============================================================
    # def eval_con(self, x):
    #     x = np.atleast_2d(x)
    #     outputs = []
    #     cons_modes = ["SA"] * self.nc  # or "SA" for surrogate
    #     if self.nc == 0:
    #         return np.empty((x.shape[0], 0))

    #     for j, mode in enumerate(cons_modes):

    #         if mode == "TRUE":
    #             gj_value = self.concept["g"][j](x)
    #             outputs.append(np.ravel(gj_value))

    #         elif mode == "SA":
    #             pred = self.PySR_con[j].predict(x)
    #             if isinstance(pred, tuple):
    #                 pred = pred[0]
    #             outputs.append(np.ravel(pred))

    #         else:
    #             raise ValueError(f"Unknown mode {mode} in cons_modes")

    #     return np.column_stack(outputs)