import numpy as np
import scipy.sparse as sp

class TermBuilderBase:
    """
    Generic builder for terms with a `.matrix()` method.
    Caches result until descriptors change.
    """
    def __init__(self, TermClass, shape: tuple, *args):
        """
        TermClass(*args, *descr) must have .matrix() → CSR of size `shape`.
        """
        self.TermClass  = TermClass
        self.args       = args
        self.shape      = shape
        self._descr     = []
        self._dirty     = True
        self._H_cached  = None

    def add(self, *descr):
        """Register one term descriptor."""
        self._descr.append(descr)
        self._dirty = True
        return self

    def build(self) -> sp.csr_matrix:
        """Assemble and cache the sum of TermClass(*args, *descr).matrix()."""
        if not self._dirty and self._H_cached is not None:
            return self._H_cached

        H = sp.csr_matrix(self.shape)
        for descr in self._descr:
            term = self.TermClass(*self.args, *descr).matrix()
            H = H + term
        self._H_cached = H.tocsr()
        self._dirty    = False
        return self._H_cached

