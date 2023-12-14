import pytest

from uegdielectric.electrongas import ElectronGas


def test_chemicalpot():
    # test that `chemicalpot` is initialized to be not `None`
    eg = ElectronGas(temperature=1, density=1)
    assert eg.chemicalpot is not None

    # Test that a warning is thrown if `DOSratio` is not `None` but `chemicalpot` is
    # `None`.
    with pytest.warns(RuntimeWarning):
        ElectronGas(temperature=1, density=1, DOSratio=lambda x: 1)
