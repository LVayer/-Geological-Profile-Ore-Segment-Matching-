import numpy as np
from stratamatch.colour_refill import refill_colours


def test_refill_precedes_instances_without_changing_observed_lithology():
    labels=np.full((80,120),-1,np.int16);domain=np.zeros(labels.shape,bool)
    domain[10:70,10:110]=True
    labels[10:70,10:60]=0;labels[10:70,60:110]=1
    labels[30:34,10:110]=-1  # annotation stroke crossing a real contact
    labels[15:25,80:100]=2  # white lithology is a known class, not an empty hole
    original=labels.copy();out,changed,audit=refill_colours(labels,domain)
    assert audit['unknown_after']==0 and changed.sum()==400
    assert np.array_equal(out[original>=0],original[original>=0])
    assert np.all(out[~domain]==-1)
    assert np.all(out[15:25,80:100]==2)
    assert np.all(out[30:34,15:50]==0) and np.all(out[30:34,70:105]==1)


def test_empty_domain_with_no_colour_seed_is_not_invented():
    labels=np.full((40,40),-1,np.int16);domain=np.ones(labels.shape,bool)
    out,_,audit=refill_colours(labels,domain)
    assert np.all(out==-1) and audit['unknown_after']==1600
