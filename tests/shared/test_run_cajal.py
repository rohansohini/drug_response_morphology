'''
Run directly: python3 tests/shared/test_run_cajal.py
'''
import os
import sys
import shutil
import tempfile
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# cajal itself needs a full conda environment and won't install in a bare
# sandbox - stub it out so run_cajal.py's top-level `from cajal import
# sample_seg` / `from cajal.qgw import quantized_gw_parallel` succeed.
# needs_recompute itself has no cajal dependency at all - this is only to
# get past the module's own imports.
_fake_cajal = types.ModuleType('cajal')
_fake_cajal.sample_seg = types.ModuleType('cajal.sample_seg')
_fake_cajal.qgw = types.ModuleType('cajal.qgw')
_fake_cajal.qgw.quantized_gw_parallel = lambda **kwargs: None
sys.modules['cajal'] = _fake_cajal
sys.modules['cajal.sample_seg'] = _fake_cajal.sample_seg
sys.modules['cajal.qgw'] = _fake_cajal.qgw

from shared.cajal.run_cajal import needs_recompute


def test_needs_recompute_is_true_when_output_is_missing():
    tmp = tempfile.mkdtemp()
    try:
        input_dir = os.path.join(tmp, 'in')
        os.makedirs(input_dir)
        open(os.path.join(input_dir, 'a.tiff'), 'w').close()

        assert needs_recompute(os.path.join(tmp, 'missing.csv'), input_dir) is True
        print("PASS: needs_recompute is True when the output file doesn't exist yet")
    finally:
        shutil.rmtree(tmp)


def test_needs_recompute_is_false_when_output_is_newer_than_every_input():
    tmp = tempfile.mkdtemp()
    try:
        input_dir = os.path.join(tmp, 'in')
        os.makedirs(input_dir)
        open(os.path.join(input_dir, 'a.tiff'), 'w').close()

        time.sleep(0.01)
        output_path = os.path.join(tmp, 'out.csv')
        open(output_path, 'w').close()

        assert needs_recompute(output_path, input_dir) is False
        print("PASS: needs_recompute is False when the output postdates every input file")
    finally:
        shutil.rmtree(tmp)


def test_needs_recompute_is_true_when_an_input_was_modified_after_the_output():
    # mirrors the real staleness scenario: extract_masks re-extracts a
    # frame (rewriting its mask TIFF) after a cached CAJAL output already
    # exists from an earlier run - the output must be recomputed, not
    # silently reused
    tmp = tempfile.mkdtemp()
    try:
        input_dir = os.path.join(tmp, 'in')
        os.makedirs(input_dir)
        stale_input = os.path.join(input_dir, 'a.tiff')
        open(stale_input, 'w').close()

        time.sleep(0.01)
        output_path = os.path.join(tmp, 'out.csv')
        open(output_path, 'w').close()

        time.sleep(0.01)
        # re-extracted: the input file gets rewritten AFTER the cached output
        with open(stale_input, 'w') as f:
            f.write('new content')

        assert needs_recompute(output_path, input_dir) is True
        print("PASS: needs_recompute is True when an input file was modified after the cached output")
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    test_needs_recompute_is_true_when_output_is_missing()
    test_needs_recompute_is_false_when_output_is_newer_than_every_input()
    test_needs_recompute_is_true_when_an_input_was_modified_after_the_output()
    print("\nAll tests passed.")
