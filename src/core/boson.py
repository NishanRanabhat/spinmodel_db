import numpy as np
import scipy.sparse as sp

class BosonMode:
    """
    Represents a single bosonic mode truncated at n_max.

    Attributes
    ----------
    n_max : int
        Maximum occupancy (i.e. the Hilbert‐space dimension is n_max + 1).
    d_b : int
        Dimension of the boson Hilbert space (n_max + 1).
    a : scipy.sparse.csr_matrix
        Annihilation operator on the boson space.
    adag : scipy.sparse.csr_matrix
        Creation operator on the boson space.
    n_op : scipy.sparse.csr_matrix
        Number operator a† a.
    H_b : scipy.sparse.csr_matrix
        On‐site boson Hamiltonian ω a†a.
    """

    def __init__(self, n_max: int):
        """
        Parameters
        ----------
        n_max : int
            Maximum boson occupancy; Hilbert space dimension = n_max + 1.
        omega : float
            Frequency of this bosonic mode.
        """
        self.n_max = n_max
        self.d_b = n_max + 1

        # Build a, a†, n_op immediately
        self.a = self._build_annihilation()
        self.adag = self.a.transpose().conj()
        self.n_op = self.adag.dot(self.a)

        # small‐op dict
        self._ops = {
            "a":    self.a,
            "adag": self.adag,
            "n":    self.n_op,
            "I":    sp.identity(self.d_b, format="csr")
        }

    def _build_annihilation(self) -> sp.csr_matrix:
        """
        Construct the (n_max+1) × (n_max+1) annihilation operator a:
            a_{i-1, i} = sqrt(i),  for i = 1..n_max
        """
        n = self.d_b
        data = []
        rows = []
        cols = []
        for i in range(1, n):
            data.append(np.sqrt(i))
            rows.append(i - 1)
            cols.append(i)
        mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        return mat

    def get_boson(self,key: str) -> sp.csr_matrix:
        assert key in self._ops, f"Unknown boson op '{key}'"
        return self._ops[key]


class BosonTerm:
    """
    Builds H = strength * boson_mode.get_small(key) (d_b×d_b).
    """
    def __init__(self, boson_mode: BosonMode, key: str, strength: float):
        self.boson_mode = boson_mode
        self.key        = key
        self.strength   = strength

    def matrix(self) -> sp.csr_matrix:
        O_b = self.boson_mode.get_boson(self.key)
        return (self.strength * O_b).tocsr()
