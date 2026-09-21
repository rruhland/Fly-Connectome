"""Exact behavior gates for the bounded memory/overhead experiments."""
import importlib.util
from pathlib import Path
import shutil

import pytest
import torch
import test_native_cpu as reference


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    path=Path(__file__).parents[1]/'scripts/benchmark_overhead_cpu.py'
    assert path.exists(), 'overhead experiment not implemented'
    spec=importlib.util.spec_from_file_location('overhead',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not shutil.which('g++'):
        pytest.skip('optional native experiment needs g++')
    library=tmp_path_factory.mktemp('overhead')/'kernel.dll'
    module.build_overhead(library)
    return module,library


@pytest.mark.parametrize('causal',[False,True])
def test_scratch_reuse_preserves_each_tick(experiment,causal):
    _,library=experiment
    reference.test_native_sparse_learning_matches_reference_each_tick(library,causal)
    reference.test_parallel_native_sparse_partitions_preserve_exact_order(library)
    # A smaller graph after a larger one exercises reused scratch bounds.
    reference.test_native_sparse_learning_matches_reference_each_tick(library,causal)


def test_native_filter_is_stable_and_does_not_alias_activity(experiment):
    module,library=experiment
    kernel=module.OverheadCPU(library)
    pathways=torch.tensor([0,1,2,0],dtype=torch.uint8)
    for edges in (torch.tensor([2,0,1,2,3,1]),torch.empty(0,dtype=torch.int64),torch.tensor([0,3])):
        saved=edges.clone()
        incoming=kernel.learning_arrivals(edges,pathways)
        assert torch.equal(incoming,edges[pathways[edges]!=0])
        incoming.fill_(0)
        assert torch.equal(edges,saved)


@pytest.mark.parametrize('eta',[.01,.5,1.2])
def test_scratch_subnormal_bits(experiment,eta):
    reference.test_native_prediction_updates_preserve_subnormal_float_bits(experiment[1],eta)


def test_filtered_learning_and_checkpoint_resume(experiment,monkeypatch,tmp_path):
    module,library=experiment
    monkeypatch.setattr(reference,'NativeCPU',module.OverheadCPU)
    reference.test_native_sparse_learning_matches_reference_each_tick(library,True)
    reference.test_native_sparse_learning_matches_reference_each_tick(library,False)
    reference.test_native_checkpoint_can_resume_on_reference_backend(library,tmp_path)
