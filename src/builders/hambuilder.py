import numpy as np
import scipy.sparse as sp
from core.spin import PauliFactory, SingleSiteTerm, TwoSiteTerm
from core.boson import BosonMode, BosonTerm
from core.spinboson import SpinBosonCouplingTerm
from core.termbuilder import TermBuilderBase

class SpinBosonModelBuilder:
    """
    Integrates spin‐only, boson‐only, and spin–boson coupling builders.
    """
    def __init__(self,
                 factory: PauliFactory,
                 boson_mode: BosonMode):
        # Sub‐builders:
        self.onesitespin_builder     = TermBuilderBase(SingleSiteTerm,
                                               (2**factory.N, 2**factory.N),
                                               factory)
        self.twositespin_builder      = TermBuilderBase(TwoSiteTerm,
                                               (2**factory.N, 2**factory.N),
                                               factory)
        self.boson_builder    = TermBuilderBase(BosonTerm,
                                               (boson_mode.d_b, boson_mode.d_b),
                                               boson_mode)
        self.coupling_builder = TermBuilderBase(SpinBosonCouplingTerm,
                                               (boson_mode.d_b*2**factory.N,
                                                boson_mode.d_b*2**factory.N),
                                               factory,
                                               boson_mode)
        self._dirty    = True
        self._H_cached = None

    # spin‐only
    def add_spin_field(self, axis: str, h: np.ndarray):
        self.onesitespin_builder.add(axis, h)
        self._dirty = True
        return self

    def add_spin_coupling(self, ops: str, J: np.ndarray):
        self.twositespin_builder.add(ops, J)
        self._dirty = True
        return self

    # boson‐only
    def add_boson_term(self, key: str, strength: float):
        self.boson_builder.add(key, strength)
        self._dirty = True
        return self

    # spin–boson
    def add_spin_boson(self, bkey: str, s_axis: str, g: float):
        self.coupling_builder.add(bkey, s_axis, g)
        self._dirty = True
        return self

    def build(self) -> sp.csr_matrix:
        if not self._dirty and self._H_cached is not None:
            return self._H_cached

        N = self.onesitespin_builder.args[0].N
        d_s = 2**N
        d_b = self.boson_builder.shape[0]

        # 1) spin‐only part: one‐site + two‐site
        H_s1 = self.onesitespin_builder.build()
        H_s2 = self.twositespin_builder.build()
        H_s = H_s1 + H_s2
        I_b = sp.identity(d_b, format="csr")
        H_spin = sp.kron(I_b, H_s, format="csr")

        # 2) boson‐only part
        H_b = self.boson_builder.build()
        I_s = sp.identity(d_s, format="csr")
        H_boson = sp.kron(H_b, I_s, format="csr")

        # 3) coupling part
        H_coup = self.coupling_builder.build()

        # 4) sum
        H_tot = (H_spin + H_boson + H_coup).tocsr()
        self._H_cached = H_tot
        self._dirty    = False
        return self._H_cached


class SpinModelBuilder:
    def __init__(self, factory: PauliFactory):
        dim = 2**factory.N
        self._onesitespin_builder = TermBuilderBase(SingleSiteTerm, (dim, dim), factory)
        self._twositespin_builder = TermBuilderBase(TwoSiteTerm,    (dim, dim), factory)
        self._dirty = True
        self._H_cached = None

    def add_spin_field(self, axis: str, h: np.ndarray):
        self._onesitespin_builder.add(axis, h)
        self._dirty = True
        return self

    def add_spin_coupling(self, ops: str, J: np.ndarray):
        self._twositespin_builder.add(ops, J)
        self._dirty = True
        return self

    def build(self):
        if not self._dirty and self._H_cached is not None:
            return self._H_cached
        H_spin = self._onesitespin_builder.build() + self._twositespin_builder.build()
        self._H_cached = H_spin.tocsr()
        self._dirty = False
        return self._H_cached
