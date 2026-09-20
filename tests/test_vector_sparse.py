"""Apply native reference regressions to the isolated staged sparse kernel."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import test_native_cpu as reference


@pytest.fixture(scope='module')
def vector_library(tmp_path_factory):
    source = Path(__file__).parents[1]/'scripts/native_vector_sparse.cpp'
    assert source.exists(), 'staged sparse experiment is not implemented'
    if not shutil.which('g++'):
        pytest.skip('optional native experiment requires g++')
    output = tmp_path_factory.mktemp('vector')/'kernels.dll'
    subprocess.run(['g++','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fopenmp',
        *(['-static'] if os.name=='nt' else []),'-static-libgcc','-static-libstdc++',
        str(source),'-o',str(output)],check=True)
    return output


@pytest.mark.parametrize('causal',[False,True])
def test_each_tick_mixed_local_learning(vector_library,causal):
    reference.test_native_sparse_learning_matches_reference_each_tick(vector_library,causal)


def test_parallel_partition_and_duplicate_order(vector_library):
    reference.test_parallel_native_sparse_partitions_preserve_exact_order(vector_library)


@pytest.mark.parametrize('eta',[.01,.5,1.,1.2])
def test_subnormal_update_bits(vector_library,eta):
    reference.test_native_prediction_updates_preserve_subnormal_float_bits(vector_library,eta)


def test_checkpoint_resume(vector_library,tmp_path):
    reference.test_native_checkpoint_can_resume_on_reference_backend(vector_library,tmp_path)
