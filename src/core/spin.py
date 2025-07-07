import numpy as np
import scipy.sparse as sp

class PauliFactory:
    """
    Lazily builds and caches all σ^α_i (α ∈ {"X","Y","Z"}, i=0..N-1)
    for an N‐site spin-½ chain.  Internally keeps a dict:

        cache[axis][i] = (2^N × 2^N) sparse CSR for σ^axis on site i.

    Only computes each (axis, i) the first time .get(axis, i) is called.
    """

    def __init__(self, N: int):

        self.N = N
        self.dim = 2**N

        # Pre‐define the 2×2 Pauli+I matrices (small, so we store them immediately):
        self._pauli2 = {
            "X": sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float64)),
            "Y": sp.csr_matrix(np.array([[0, -1j], [1j,  0]], dtype=np.complex128)),
            "Z": sp.csr_matrix(np.array([[1,  0], [0, -1]], dtype=np.float64)),
            "+": sp.csr_matrix(np.array([[0,  1], [0, 0]], dtype=np.float64)),
            "-": sp.csr_matrix(np.array([[0,  0], [1, 0]], dtype=np.float64))
        }
        self._I2 = sp.identity(2, format="csr", dtype=np.float64)

        # Prepare empty caches for "X","Y","Z","+","-"
        self.cache = {
            "X": {},    # will map i -> σ^X_i
            "Y": {},    # will map i -> σ^Y_i
            "Z": {},     # will map i -> σ^Z_i
            "+": {},     # will map i -> +_i
            "-": {}     # will map i -> -_i
        }

    def get(self, axis: str, site: int) -> sp.csr_matrix:
        """
        Return σ^axis at site index `site`, as a sparse CSR of shape (2^N × 2^N).
        Builds it the first time and caches in self.cache[axis][site].

        Parameters
        ----------
        axis : str, one of "X", "Y", "Z"
        site : int, in [0 .. N-1]
        """
        assert axis in {"X", "Y", "Z", "+", "-"}, "axis must be 'X','Y','Z','+', or '-'"
        assert 0 <= site < self.N, "site must be between 0 and N-1"

        # If it’s already cached, return it immediately
        if site in self.cache[axis]:
            return self.cache[axis][site]

        # Otherwise, build it now:
        pauli2 = self._pauli2[axis]
        op = None
        for i in range(self.N):
            factor = pauli2 if (i == site) else self._I2
            op = factor if (op is None) else sp.kron(op, factor, format="csr")

        # Cache, then return
        self.cache[axis][site] = op.tocsr()
        return self.cache[axis][site]

class SingleSiteTerm:
    """
    Represents H = Σ_{i=0..N-1} h[i] * σ^axis_i.
    """

    def __init__(self,
                 factory: PauliFactory,
                 axis: str,
                 h: np.ndarray):
        assert h.ndim == 1 and h.shape[0] == factory.N
        assert axis in {"X", "Y", "Z", "+", "-"}
        self.factory = factory
        self.h = h.copy()
        self.axis = axis

    """
    Builds a sparse matrix out of the input arguments
    """
    def matrix(self) -> sp.csr_matrix:
        N = self.factory.N
        H = None
        for i, coeff in enumerate(self.h):
            if coeff == 0:
                continue
            sigma = self.factory.get(self.axis, i)
            term = sigma.multiply(coeff)
            H = term if H is None else (H + term)

        if H is None:
            # all h[i] == 0 → zero matrix
            return sp.csr_matrix((2**N, 2**N))
        return H.tocsr()

class TwoSiteTerm:

    def __init__(self,
                 factory: PauliFactory,
                 ops: str,
                 J: np.ndarray):
        N = factory.N
        assert J.shape == (N, N)
        assert isinstance(ops, str) and len(ops) == 2
        assert ops[0] in {"X", "Y", "Z", "+", "-"} and ops[1] in {"X", "Y", "Z", "+", "-"}

        self.factory = factory
        self.J = J.copy()
        self.ops = ops 

    def matrix(self) -> sp.csr_matrix:

        N = self.factory.N
        H = None

        for i in range(N-1):
            sigma_i = self.factory.get(self.ops[0],i)
            for j in range(i+1,N):
                coeff = self.J[i,j]
                if coeff == 0:
                    continue
                sigma_j = self.factory.get(self.ops[1],j)
                term = sigma_i.dot(sigma_j).multiply(coeff)
                H = term if H is None else (H + term)

        if H is None:
            # all h[i] == 0 → zero matrix
            return sp.csr_matrix((2**N, 2**N))
        return H.tocsr()

