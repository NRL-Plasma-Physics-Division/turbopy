"""Integration test based on block_on_spring turbopy app

This integration test simply runs the block_on_spring app and compares the output files
to "good" output.
"""
import numpy as np


def test_bos(bos_run):
    """Tests block_on_spring app with ForwardEuler ComputeTool and compares to
    output files with a "good" output
    """

    for filename in ['block_p', 'block_x', 'time']:
        ref_data = np.genfromtxt(f'tests/fixtures/block_on_spring/output/{filename}.csv',
                                 delimiter=',')
        tmp_data = np.genfromtxt(f'tmp/block_on_spring/output/{filename}.csv',
                                 delimiter=',')
        assert np.allclose(ref_data, tmp_data, rtol=1e-05, atol=1e-08)
