"""Refill cleared annotations inside the fixed outer domain before layer extraction."""
import numpy as np
from .staged_reconstruction import _repair_linear_gaps, _repair_regions


def refill_colours(labels,domain,config=None):
    """Keep observed classes fixed; reconstruct missing colours and audit uncertainty.

    Narrow gaps are repaired first, then remaining regions are partitioned by
    surrounding classes. The outer mask is a hard limit, including its concavities.
    Filling is an explicit reconstruction hypothesis, not new observed evidence.
    """
    cfg=config or {};before=labels.copy();before[~domain]=-1
    observed=(before>=0)&domain
    linear,linear_audit=_repair_linear_gaps(before,domain,int(cfg.get('refill_line_probe_pixels',12)))
    linear[observed]=before[observed];linear[~domain]=-1
    filled,region_audit=_repair_regions(linear,domain)
    filled[observed]=before[observed];filled[~domain]=-1
    changed=(before<0)&(filled>=0)&domain
    assert np.array_equal(filled[observed],before[observed])
    assert np.all(filled[~domain]<0)
    return filled,changed,{'method':'linear_gap_then_local_region_partition',
        'linear':linear_audit,'regions':region_audit,
        'unknown_before':int(np.count_nonzero((before<0)&domain)),
        'refilled_pixels':int(changed.sum()),
        'unknown_after':int(np.count_nonzero((filled<0)&domain)),
        'observed_class_changes':0,'filled_outside_domain':0,
        'interpretation':'reconstructed colours; white and multi-class gaps require review'}
