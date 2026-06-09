import itertools


class Namer:
    all_names = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]

    def __init__(self):
        self.gen = itertools.count()

    def get_name(self):
        idx = next(self.gen)
        base = self.all_names[idx % len(self.all_names)]
        suffix = idx // len(self.all_names)
        return base if suffix == 0 else f"{base}_{suffix + 1}"
