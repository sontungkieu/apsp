"""Gate decision rules (Spec §11.3-11.4). Agents may compute and recommend; research approves claims."""
from __future__ import annotations

from dataclasses import dataclass

GO, STOP, INC = "GO", "STOP", "INC"
COST_GO_SAVINGS = 0.10
COST_STOP_SAVINGS = 0.05


@dataclass(frozen=True)
class Thresholds:
    delta_anchor: float
    tau_prac: float
    theta_H: float
    theta_q: float
    delta_IR: float
    theta_RL: float
    theta_pay: float


def thresholds(delta_anchor: float) -> Thresholds:
    """tau = max(0.2*Delta_anchor, 0.02); theta_H = tau; theta_q = delta_IR = tau/2; theta_RL = theta_pay = tau/4."""
    tau = max(0.2 * delta_anchor, 0.02)
    return Thresholds(delta_anchor, tau, tau, tau / 2, tau / 2, tau / 4, tau / 4)


def quality_decision(lcb: float, ucb: float, theta: float) -> str:
    """GO if LCB >= theta; STOP (for that practical effect) if UCB < theta; else INC."""
    if lcb >= theta:
        return GO
    if ucb < theta:
        return STOP
    return INC


def ni_pass(lcb_quality_diff: float, delta_ir: float) -> bool:
    """Non-inferiority: LCB(Y_m - Y_B2) > -delta_IR."""
    return lcb_quality_diff > -delta_ir


def cost_decision(lcb_quality_diff: float, ucb_quality_diff: float, lcb_savings: float, ucb_savings: float,
                  delta_ir: float) -> str:
    """Cost gate with distinct thresholds: GO needs NI and LCB(savings) >= 10%;
    STOP if UCB(savings) < 5% or UCB(quality diff) < -delta_IR; the 5-10% grey zone stays INC."""
    if ucb_savings < COST_STOP_SAVINGS or ucb_quality_diff < -delta_ir:
        return STOP
    if ni_pass(lcb_quality_diff, delta_ir) and lcb_savings >= COST_GO_SAVINGS:
        return GO
    return INC


def after_second_look(decision_look2: str) -> str:
    """INC after look 2 is reported as 'inconclusive', never converted into evidence of no effect."""
    return "inconclusive" if decision_look2 == INC else decision_look2


def project_status(g2: str, all_primary_caps_breadth_dominated: bool, g3_branches: list[str]) -> str:
    if g2 == STOP or all_primary_caps_breadth_dominated or (g3_branches and all(b == STOP for b in g3_branches)):
        return STOP
    if any(b == GO for b in g3_branches):
        return GO
    return INC
