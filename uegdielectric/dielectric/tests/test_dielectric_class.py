from uegdielectric.dielectric.dielectric_class import Mermin
from uegdielectric.electrongas import ElectronGas
import pytest
from typing import Callable
from numbers import Number


class TestMerminConstruction:
    """Tests for different ways to initialize a Mermin object."""

    @classmethod
    def setup_class(cls):
        # create an electrongas instance
        cls.elecgas = ElectronGas(1, 1)

    def test_collfreq_none(self):
        """Test when there is no argument for `collfreq`."""
        m = Mermin(self.elecgas)
        errstr = (
            "A Mermin instance initialized without an argument for `collfreq` should "
            + "yield a constant (0) function for `collfreq`."
        )
        assert m.collfreq(42) == pytest.approx(0), errstr

    @pytest.mark.parametrize("collfreq_in", [2, 42.0, 1 + 1j])
    def test_collfreq_const(self, collfreq_in: Number):
        """Test when the argument for `collfreq` is a constant."""
        collfreq_in = 42.0
        m = Mermin(self.elecgas, collfreq_in)
        errstr = (
            "A Mermin instance initialized with a constant `collfreq` should "
            + "yield a constant function for `collfreq`."
        )
        assert m.collfreq(1.33) == pytest.approx(collfreq_in), errstr

    def test_collfreq_func(self):
        """Test when the argument for `collfreq` is a function of one variable."""
        collfreq_in = lambda x: 2 * x
        m = Mermin(self.elecgas, collfreq_in)
        x = 1.33
        errstr = (
            "A Mermin instance initialized with a function for `collfreq` should"
            + " yield that function for `collfreq`."
        )
        assert m.collfreq(x) == pytest.approx(collfreq_in(x)), errstr


@pytest.mark.parametrize(
    "collfreq_init, collfreq_new",
    [
        (1, None),
        (lambda x: x, 1),
        (None, 42.0),
        (1.33, 1 + 1j),
        (2j, lambda x: x**2),
    ],
)
def test_collfreq_setter(
    collfreq_init: Number | Callable[[int | float], Number] | None,
    collfreq_new: Number | Callable[[int | float], Number] | None,
):
    """Test for the collfreq setter function, which has same functionality when
    creating a Mermin object."""
    elecgas = ElectronGas(0.01, 1e-3)
    m = Mermin(elecgas, collfreq_init)
    m.collfreq = collfreq_new
    if collfreq_new is None:
        assert m.collfreq(0.0) == pytest.approx(0.0)
    elif isinstance(collfreq_new, Number):
        assert m.collfreq(42) == pytest.approx(collfreq_new)
    else:
        assert m.collfreq == collfreq_new
