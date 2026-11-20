"""VERL integration (GPU stack). Only :mod:`.advantage`'s tensor function is exercised in CPU CI."""

from agentic_search.integrations.verl.advantage import (
    ESTIMATOR_NAME,
    AdvantageConfig,
    compute_gspo_gigpo_opd_advantage,
    install_verl_hooks,
)

__all__ = ["ESTIMATOR_NAME", "AdvantageConfig", "compute_gspo_gigpo_opd_advantage", "install_verl_hooks"]
