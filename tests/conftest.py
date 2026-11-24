import pytest

from wellprod.bench import load
from wellprod.synth import SynthConfig

SMALL = SynthConfig(n_fields=3, pads_per_field=(5, 7), wells_per_pad=(3, 5))


@pytest.fixture(scope="session")
def small_basin():
    return load(11, SMALL)
