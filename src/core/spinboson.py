import numpy as np
import scipy.sparse as sp
from .spin import PauliFactory, SingleSiteTerm
from .boson import BosonMode

class SpinBosonCouplingTerm:
    """
    Builds H = g * (O_b ⊗ Σ_i σ^spin_axis_i).
    """
    def __init__(self,
                 factory: PauliFactory,
                 boson_mode: BosonMode,
                 bkey: str,
                 spin_axis: str,
                 g: float):
        self.factory   = factory
        self.boson_mode= boson_mode
        self.bkey      = bkey
        self.spin_axis = spin_axis
        self.g         = g

    def matrix(self) -> sp.csr_matrix:
        # Boson operator:
        O_b = self.boson_mode.get_boson(self.bkey)
        # Global spin operator Σ_i σ^spin_axis_i:
        h_all = np.ones(self.factory.N)
        O_s   = SingleSiteTerm(self.factory, self.spin_axis, h_all).matrix()
        # Tensor and scale:
        return (sp.kron(O_b, O_s, format="csr") * self.g).tocsr()
