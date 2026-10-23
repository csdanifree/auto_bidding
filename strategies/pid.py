import numpy as np
from auto_bidding.strategies.base import BaseBiddingStrategy


class PidCpaBiddingStrategy(BaseBiddingStrategy):
    """
      error_t = target_cpa - real_cpa_t
      alpha_{t+1} = alpha_t + Kp * error_t + Ki * sum(error) + Kd * (error_t - error_{t-1})
    """

    def __init__(
        self,
        budget=100,
        name="PID-CPA-Strategy",
        cpa=2,
        category=1,
    ):
        super().__init__(budget, name, cpa, category)

        self.kp = 0.1
        self.ki = 0.01
        self.kd = 0.03
        self.alpha_min = 0.01
        self.alpha_max = float(cpa * 3)
        self.integral_clip = 50.0
        self.ema_ratio = 0.5
        self.min_conv = 1e-6
        self.reset()

    def reset(self):
        self.remaining_budget = self.budget
        self._error_sum = 0.0
        self._error_prev = 0.0
        self._cum_cost = 0.0
        self._cum_conv = 0.0
        self._cum_pvalue = 0.0
        self._cpa_ema = None
        self._alpha = self.cpa

    def _get_last_tick_stats(self, historyAuctionResult, historyImpressionResult):
        cost, conv = 0.0, 0.0
        if historyAuctionResult and len(historyAuctionResult) > 0:
            last = historyAuctionResult[-1]
            if last is not None and len(last) > 0:
                cost = float(np.sum(last[:, 2]))
        if historyImpressionResult and len(historyImpressionResult) > 0:
            last = historyImpressionResult[-1]
            if last is not None and len(last) > 0:
                conv = float(np.sum(last[:, 0]))
        return cost, conv

    def _compute_real_cpa(self):
        if self._cum_conv < self.min_conv:
            if self._cum_pvalue > self.min_conv:
                return self._cum_cost / self._cum_pvalue
            return self.cpa
        return self._cum_cost / self._cum_conv

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
        if self.remaining_budget <= 0:
            return np.zeros_like(pValues), 0.0

        self._cum_pvalue += float(np.sum(pValues))

        if timeStepIndex > 0:
            cost, conv = self._get_last_tick_stats(historyAuctionResult, historyImpressionResult)
            self._cum_cost += cost
            self._cum_conv += conv

        real_cpa = self._compute_real_cpa()
        
        if self._cpa_ema is None:
            self._cpa_ema = real_cpa
        else:
            self._cpa_ema = self.ema_ratio * real_cpa + (1 - self.ema_ratio) * self._cpa_ema

        error = self.cpa - self._cpa_ema
        
        cpa_ratio = self._cpa_ema / self.cpa if self.cpa > 0 else 1.0
        
        if cpa_ratio > 1.3:
            boost = 1.0 + (cpa_ratio - 1.0)
        elif cpa_ratio < 0.7:
            boost = 1.0 + (1.0 - cpa_ratio)
        else:
            boost = 1.0
        
        if timeStepIndex > 0:
            self._error_sum = np.clip(self._error_sum + error, -self.integral_clip, self.integral_clip)
            error_diff = error - self._error_prev
            delta = boost * (self.kp * error + self.ki * self._error_sum + self.kd * error_diff)
            self._alpha += delta
            self._error_prev = error

        self._alpha = np.clip(self._alpha, self.alpha_min, self.alpha_max)
        bids = self._alpha * pValues

        return bids, self._alpha