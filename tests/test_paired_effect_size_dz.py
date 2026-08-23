"""Regression: paired comparisons must report Cohen's d_z, not pooled d.

`ttest_comparison(..., paired=True)` and `wilcoxon_comparison(...)` (always
paired) both reported `effect_size` via the pooled independent-groups formula
even though the test statistic is computed on paired/matched samples. The
correct paired effect size is d_z: mean(differences) / std(differences, ddof=1).
Issue #2154.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.utils.statistics import (
    cohens_d,
    ttest_comparison,
    wilcoxon_comparison,
)

pytestmark = pytest.mark.unit

# Every pair shows a consistent +1 shift; between-seed variance swamps the
# pooled denominator, so pooled d looks negligible while d_z is large.
A = np.array([101.0, 202.0, 303.0, 404.0])
B = np.array([100.0, 200.0, 300.0, 400.0])
D_Z = float(np.mean(A - B) / np.std(A - B, ddof=1))


def test_paired_ttest_reports_dz_not_pooled_d() -> None:
    result = ttest_comparison(A, B, paired=True)
    assert result.effect_size == pytest.approx(D_Z)
    # The pooled formula would report ~0.019; d_z is ~1.94.
    assert result.effect_size != pytest.approx(cohens_d(A, B))


def test_unpaired_ttest_still_reports_pooled_d() -> None:
    result = ttest_comparison(A, B, paired=False)
    assert result.effect_size == pytest.approx(cohens_d(A, B))


def test_wilcoxon_reports_dz() -> None:
    result = wilcoxon_comparison(A, B)
    assert result.effect_size == pytest.approx(D_Z)


def test_paired_dz_sign_convention() -> None:
    # Positive means a > b, matching cohens_d's convention.
    assert ttest_comparison(A, B, paired=True).effect_size > 0
    assert ttest_comparison(B, A, paired=True).effect_size < 0
    assert wilcoxon_comparison(A, B).effect_size > 0


def test_paired_dz_zero_variance_differences() -> None:
    # Constant nonzero differences: std(d)=0 -> signed-infinite d_z,
    # mirroring cohens_d's zero-pooled-std convention.
    a = np.array([2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0])
    assert ttest_comparison(a, b, paired=True).effect_size == float("inf")
    assert ttest_comparison(b, a, paired=True).effect_size == float("-inf")
