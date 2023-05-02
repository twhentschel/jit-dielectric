"""Tests the inv_fdint_onehalf function"""

import numpy as np

from uegdielectric.electrongas._inv_fermi_integral import inv_fdint_onehalf


class Test_inv_fdint_onehalf:
    @classmethod
    def setup_class(cls):
        """
        Initialize parameters for tests.

        Testing the degenerate (:math:`\eta >> 1`) (where
        :math:`\eta` = [chemical potential] / [thermal energy]), non-degenerate
        (:math:`\eta << 1`), and somewhere in-between (:math:`\eta \approx 1')
        cases.

        Compare inverse to forward solve, which amounts to evaluating the
        Fermi-Dirac integral of order 1/2. This is related to the polylogarithm of 3/2
        :math:`\mathrm{Li}_{3/2}`, via the formula

        ..math::
        \mathrm{density} = \mathrm{Li}_{3/2}(-e^{\eta})
        \frac{k_B T}{\sqrt{2} \pi^{3/2}}.

        """
        cls.temp = [1e-3, 1 / 27.2114, 10 / 27.2114]
        cls.chempot = [0.42851298, 0.0321254, -0.0294728]
        cls.den = [0.0267962, 0.00129713, 0.0203079]

    def test_known1(self):
        """
        Test inv_fdint_onehalf for a known value with :math:`eta >> 1`
        """
        methodval = inv_fdint_onehalf(self.den[0], self.temp[0])
        testTrue = np.isclose(methodval, self.chempot[0], rtol=0.0, atol=1e-4)
        errStr = (
            "Degenerate case: inverse Fermi integral (1/2) value should be "
            + f"approximately {self.chempot[0]} and not {methodval}."
        )
        assert testTrue, errStr

    def test_known2(self):
        """
        Test inv_fdint_onehalf for a known value with :math:`eta \approx 1`
        """
        methodval = inv_fdint_onehalf(self.den[1], self.temp[1])
        testTrue = np.isclose(methodval, self.chempot[1], rtol=0.0, atol=1e-4)
        errStr = (
            "Partially-degenerate case: inverse Fermi integral (1/2) value "
            + f"should be approximately {self.chempot[1]} and not {methodval}."
        )
        assert testTrue, errStr

    def test_known3(self):
        """
        Test inv_fdint_onehalf for a known value with :math:`eta << 1`
        """
        methodval = inv_fdint_onehalf(self.den[2], self.temp[2])
        testTrue = np.isclose(methodval, self.chempot[2], rtol=0.0, atol=1e-4)
        errStr = (
            "Non-degenerate case: inverse Fermi integral (1/2) value should be "
            + f"approximately {self.chempot[2]} and not {methodval}."
        )
        assert testTrue, errStr
