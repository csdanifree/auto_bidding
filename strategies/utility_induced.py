import numpy as np

from auto_bidding.strategies.base import BaseBiddingStrategy


class ScoreMarginalBiddingStrategy(BaseBiddingStrategy):
    """Utility-induced bidding with fixed dense/sparse configurations.

    Both variants use beta=0, gamma=1.8, tau_rel=0.005 and forecast
    conversions for the utility marginal and CPA feedback. Duals update
    every timestep. Only the fixed step size differs: sparse=0.01,
    dense=0.0001. The evaluator sets ``_total_steps`` to the episode length
    and updates ``remaining_budget`` before each call to ``bidding``.

    The three utility pieces remain (1, 0), (1, 0), (2.8, -1.8).
    Soft-min weights determine the marginal coefficients (a_t, d_t), then
    bids are cpa * pValue * (a_t + lambda) / (lambda + mu - d_t),
    capped by the bid limit and available budget. A nonpositive
    denominator selects the cap.
    """

    _STEP_SIZES = {"sparse": 1.0e-2, "dense": 1.0e-4}

    def __init__(
        self,
        budget=100,
        name="Score-Marginal-Bidding",
        cpa=2,
        category=1,
        variant="dense",
    ):
        if variant not in self._STEP_SIZES:
            raise ValueError("variant must be 'dense' or 'sparse'")
        super().__init__(budget, name, cpa, category)
        self.variant = variant
        self.beta = 0.0
        self.gamma = 1.8
        self.tau_rel = 0.005
        self.U = 1.0
        self.bar_b = 1.0e6
        self.eta_d = self._STEP_SIZES[variant]
        self.n_ref = 1.0
        self.M = 1.0e3
        self._a_pieces = np.array([1.0, 1.0, 1.0 + self.gamma], dtype=np.float64)
        self._d_pieces = np.array([0.0, 0.0, -self.gamma], dtype=np.float64)
        self._total_steps = None
        self.reset()

    def reset(self):
        self.remaining_budget = self.budget
        self.N = 0.0
        self.P_cash = 0.0
        self.lam = 0.0
        self.mu = 1.0
        self._P_at_bnd = 0.0
        self._N_at_bnd = 0.0

    def _get_last_tick_expected(self, historyAuctionResult, historyPValueInfo):
        """Read cash payment and sum(pValue * win) from the previous tick."""
        cost, expected_conv = 0.0, 0.0
        if historyAuctionResult and len(historyAuctionResult) > 0:
            last = historyAuctionResult[-1]
            if last is not None and len(last) > 0:
                last = np.asarray(last)
                cost = float(np.sum(last[:, 2]))
                win = last[:, 0].astype(np.float64)
                if historyPValueInfo and len(historyPValueInfo) > 0:
                    previous_values = historyPValueInfo[-1]
                    if previous_values is not None and len(previous_values) > 0:
                        pvalues = np.asarray(previous_values)[:, 0].astype(np.float64)
                        n = min(len(pvalues), len(win))
                        expected_conv = float(np.sum(pvalues[:n] * win[:n]))
        return cost, expected_conv

    def _update_duals(self):
        """Update both duals from the immediately preceding timestep."""
        p_inc = self.P_cash - self._P_at_bnd
        n_inc = self.N - self._N_at_bnd
        total_steps = self._total_steps if self._total_steps and self._total_steps > 0 else 1
        p_norm = p_inc / float(self.cpa)
        rho_norm = float(self.budget) / float(total_steps) / float(self.cpa)
        lam_grad = (p_norm - self.U * n_inc) / self.n_ref
        mu_grad = (p_norm - rho_norm) / self.n_ref
        self.lam = float(np.clip(self.lam + self.eta_d * lam_grad, 0.0, self.M))
        self.mu = float(np.clip(self.mu + self.eta_d * mu_grad, 0.0, self.M))
        self._P_at_bnd = self.P_cash
        self._N_at_bnd = self.N

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
        actual_excuted_action=None,
    ):
        pValues = np.asarray(pValues, dtype=np.float32)
        available_budget = float(self.remaining_budget)
        if available_budget <= 0.0:
            return np.zeros_like(pValues), float(self.lam)

        if int(timeStepIndex) > 0:
            cost, expected_conv = self._get_last_tick_expected(
                historyAuctionResult, historyPValueInfo
            )
            self.N += expected_conv
            self.P_cash += cost
            self._update_duals()

        c = float(self.cpa)
        ratio = (self.P_cash / c + 2.0) / (self.N + 2.0)
        pieces = self._a_pieces + self._d_pieces * ratio
        pieces = pieces - pieces.min()
        weights = np.exp(-pieces / self.tau_rel)
        weights = weights / weights.sum()
        a_t = float(np.sum(weights * self._a_pieces))
        d_t = float(np.sum(weights * self._d_pieces))

        denominator = self.lam + self.mu - d_t
        cap = min(self.bar_b, available_budget)
        if denominator > 0.0:
            raw_bids = c * pValues.astype(np.float64) * (a_t + self.U * self.lam) / denominator
            bids = np.minimum(cap, raw_bids)
        else:
            bids = np.full(pValues.shape, cap, dtype=np.float64)
        return np.maximum(bids, 0.0).astype(np.float32), float(self.lam)