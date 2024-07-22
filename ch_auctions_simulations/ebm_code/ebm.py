import itertools

import numpy as np


# we use simple integral and don't calculate interim utility
class ExclusiveBuyerMechanismApproximation:

    def __init__(self, N, X, c, T, f):
        self.N = int(N)
        self.X = list(X)
        self.J = len(X)
        self.c = list(c)
        self.T = int(T)
        self.f = f

        assert len(c) == len(X)
        assert self.J <= 2, "can't handle more then 2 quality grades"

        Xj_deltas = [Xj[1] - Xj[0] for Xj in X]
        assert len(np.unique(Xj_deltas)) == 1, "rectangles not supported"
        self.delta_1d = Xj_deltas[0] / T

        self.Xj_ranges = []
        for Xj in self.X:
            Xj_range = np.arange(Xj[0], Xj[1] + 1e-10, self.delta_1d)
            self.Xj_ranges.append(Xj_range)
        self.X_iter = list(itertools.product(*self.Xj_ranges))

        self.ff = f
        self._FNminus1 = self._precompute_FNminus1()
        self.total_mass = self._FNminus1[len(self.X_iter)-1]

    def _precompute_FNminus1(self):
        FNMinus1 = dict()
        for i, x in enumerate(self.X_iter):
            FNMinus1[i] = self._FNminus1_raw(x)
        return FNMinus1

    def _FNminus1_raw(self, x):
        # TODO precompute this and store result
        if self.N == 1:
            return 1

        else:  # N > 1
            Xj_subsets = []
            for j in range(self.J):
                Xj_subset = self.Xj_ranges[j][
                    self.Xj_ranges[j] <= x[j] + 1e-10]
                Xj_subsets.append(Xj_subset)
            X_subset = itertools.product(*Xj_subsets)

            subset_mass = 0
            for i, x in enumerate(X_subset):
                subset_mass += self.ff(x) * np.power(self.delta_1d, self.J) * \
                    self.numerical_adjustment(i)

            return np.power(subset_mass, self.N - 1)

    def FNminus1(self, x):
        ixs = []
        for j in range(self.J):
            ix = np.where(np.isclose(self.Xj_ranges[j], x[j]))[0][0]
            ixs.append(ix)

        i = ixs[0] * (self.T + 1) + ixs[1]
        return self._FNminus1[i]

    def allocation(self, x, p):
        x = list(x)
        betas = [x[j] - p[j] for j in range(self.J)]

        if self.J == 1:
            F = self.FNminus1(x)
            return (int(x[0] - p[0] >= 0 - 1e-10) * F, )

        else:  # J == 2

            if np.isclose(*betas) and betas[0] >= 0 - 1e-10:
                Q1 = Q2 = 1/2
            else:
                Q1 = int(betas[0] > betas[1]) * int(betas[0] >= 0 - 1e-10)
                Q2 = int(betas[1] > betas[0]) * int(betas[1] >= 0 - 1e-10)

            # avoid computing FNminus1 with beta < 0
            if Q1 > 0:
                F1 = self.FNminus1(
                    [x[0], min(max(self.Xj_ranges[1]), p[1] + betas[0])])
            else:
                F1 = 0
            if Q2 > 0:
                F2 = self.FNminus1(
                    [min(max(self.Xj_ranges[0]), p[0] + betas[1]), x[1]])
            else:
                F2 = 0

            return (Q1 * F1, Q2 * F2)

    def numerical_adjustment(self, i):
        adjustment = 1
        if self.J == 1:
            if i in [0, self.T]:
                adjustment /= 2
        else:  # J == 2
            j1 = int(i / (self.T+1))
            j2 = i % (self.T+1)
            if j1 in [0, self.T]:
                adjustment /= 2
            if j2 in [0, self.T]:
                adjustment /= 2
        return adjustment

    def obj(self, p):
        p = list(p)
        assert len(p) == self.J, "len(p) != J"
        self.Q = []
        self.total = 0
        for i, x in enumerate(self.X_iter):
            Qs = self.allocation(x, p)
            sum_j = 0
            for j in range(self.J):
                sum_j += Qs[j] * (p[j] - self.c[j])
            revenue = self.N * sum_j * self.ff(x) * \
                np.power(self.delta_1d, self.J) * self.numerical_adjustment(i)

            self.total += revenue
            self.Q.append(Qs)
        return self.total, self.Q

    def evaluate(self, ps, progress=True):
        n_prices = len(ps)
        inc = int(n_prices / 10)
        results = []
        for i, p in enumerate(ps):
            if i % inc == 0 and progress:
                print("%s/%s" % (i, n_prices))
            results.append(self.obj(p)[0])
        return results
