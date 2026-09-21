import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('scipy')
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from multitempo import training_schedule


def test_interleaved_training_excludes_held_out_tempo_and_preserves_events():
    frames,blanks,dwells=training_schedule(12,9060)
    assert set(dwells)=={2,4,6}
    assert np.array_equal(np.unique(dwells,return_counts=True)[1],[4,4,4])
    assert len(frames)==sum(blanks+8*dwells+1)
    signal=frames[:,0,30,39].numpy().astype(int)
    delta=np.diff(signal,prepend=0)
    assert (delta==1).sum()==(delta==-1).sum()==48
    again=training_schedule(12,9060)
    np.testing.assert_array_equal(blanks,again[1])
    np.testing.assert_array_equal(dwells,again[2])
