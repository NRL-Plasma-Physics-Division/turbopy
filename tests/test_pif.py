"""Integration test based on particle_in_field turbopy app

This integration test simply runs the particle_in_field app and compares the output files
to "good" output.
"""
import numpy as np

def test_pif(pif_run):
    for filename in ['e_0.5', 'grid', 'particle_p', 'particle_x', 'time']:
        ref_data = np.genfromtxt(f'tests/fixtures/particle_in_field/output/{filename}.csv',
                              delimiter=',')
        tmp_data = np.genfromtxt(f'tmp/particle_in_field/output/{filename}.csv',
                              delimiter=',')
        assert (np.allclose(ref_data, tmp_data, rtol=1e-05, atol=1e-08))
