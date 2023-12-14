import numpy as np

import pytest

from uegdielectric.dielectric.dielectric_class import Mermin, RPA
from uegdielectric.electrongas import ElectronGas


class TestMerminConstruction:
    """Tests for different ways to initialize a Mermin object."""

    @classmethod
    def setup_class(cls):
        # create an electrongas instance
        cls.elecgas = ElectronGas(1, 1)

    def test_collisionrate_none(self):
        """Test when there is no argument for `collisionrate`."""
        m = Mermin(self.elecgas)
        errstr = (
            "A Mermin dielectric function called without a value for `collisionrate`"
            + " should be the same as letting `collisionrate` be a function that"
            + " returns 0."
        )
        assert m(1.0, 3.11) == pytest.approx(m(1.0, 3.11, lambda x: 0)), errstr

    @pytest.mark.parametrize("m, n", [(1, 10), (10, 1), (10, 10)])
    def test_Mermin_multiarray_input(self, m, n):
        """Testing output shapes of Mermin dielectric function."""
        q = np.linspace(0.1, 1, m)
        omega = np.linspace(0, 1, n)
        eps = Mermin(self.elecgas)
        out = eps(q, omega)

        assert (m, n) == out.shape

    # ignore warnings for evaluating the Mermin dielectric at `wavenum` = 0.
    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize("arg", ["wavenum", "frequency"])
    def test_Mermin_singlearray_input(self, arg):
        n = 9
        if arg == "wavenum":
            eps = Mermin(self.elecgas)(1.1, np.linspace(0, 1, n))
            assert (9,) == eps.shape
        else:
            eps = Mermin(self.elecgas)(np.linspace(0, 1, n), 3.14)
            assert (9,) == eps.shape


class TestRPA:
    """Tests for the initialization and use of the RPA model."""

    elecgas = ElectronGas(1.618, 3.141)

    def test_rpa(self):
        """Test that RPA initilization is same as Mermin with collisionrate=None"""
        r = RPA(self.elecgas)
        m = Mermin(self.elecgas)
        assert r(1, 1) == pytest.approx(m(1, 1))
