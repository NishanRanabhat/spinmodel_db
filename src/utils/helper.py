import numpy as np
import sqlite3
import json
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from core.spin import PauliFactory
from builders.hambuilder import SpinModelBuilder
from db.database import SpectrumDatabase

def build_spin_hamiltonian(params: dict):
    """
    params should contain:
      • "N"   : int, number of sites
      • "JXX", "JYY", "JZZ":  N×N coupling matrices (or missing/None)
      • "hX", "hY", "hZ":  length-N field arrays (or missing/None)

    Returns the sparse Hamiltonian (csr_matrix).
    """
    # 1) Required parameter
    N = params.get("N")
    if N is None:
        raise ValueError("Missing required parameter 'N'")
    
    # 2) Initialize factory & builder
    factory = PauliFactory(N)
    builder = SpinModelBuilder(factory)

    # 3) Add two-site couplings if provided
    for key, ops in [("JXX", "XX"), ("JYY", "YY"), ("JZZ", "ZZ")]:
        J = params.get(key)
        if J is not None and np.any(J):
            J = np.asarray(J)
            if J.shape != (N, N):
                raise ValueError(f"Parameter '{key}' must be shape ({N},{N})")
            print(ops)
            builder.add_spin_coupling(ops, J)

    # 4) Add single-site fields if provided
    for key, axis in [("hX","X"), ("hY","Y"), ("hZ","Z")]:
        h = params.get(key)
        if h is not None and np.any(h):
            h = np.asarray(h)
            if h.shape != (N,):
                raise ValueError(f"Parameter '{key}' must have length {N}")
            print(axis)
            builder.add_spin_field(axis, h)

    # 5) Build & return
    return builder.build()

def process_runs(params_list, path="spectra.db", dense_threshold=12, sparse_k=6):
    """
    Loop over parameter dicts, build and diagonalize each Hamiltonian,
    use dense diagonalization for small N, sparse for larger N,
    and store the resulting eigenvalues in the SQLite database.

    params_list : list of dict
        Each dict must include 'N' and optional 'XX','YY','ZZ','HX','HY','HZ'.
    path : str
        Path to the SQLite database file.
    dense_threshold : int
        Maximum N for which to use dense diagonalization.
    sparse_k : int
        Number of smallest eigenvalues to compute when using sparse diagonalization.
    """
    # Initialize database
    db = SpectrumDatabase(path)

    for params in params_list:
        N = params.get("N")
        if N is None:
            raise ValueError("Each params dict must include 'N'")

        # Build Hamiltonian
        H_spin = build_spin_hamiltonian(params)
        dim = H_spin.shape[0]

        # Diagonalize
        if N <= dense_threshold:
            # full spectrum
            H_dense = H_spin.toarray()
            eigvals, eigvecs = np.linalg.eigh(H_dense)
            method = 'dense'
        else:
            # sparse: few lowest modes
            k = min(sparse_k, dim - 2)
            eigvals, eigvecs = spla.eigsh(H_spin, k=k, which='SM')
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            method = 'sparse'

        # Ensure params dict is JSON-serializable
        #serial_params = {}
        #for key, val in params.items():
        #    if isinstance(val, np.ndarray):
        #        serial_params[key] = val.tolist()
        #    else:
        #        serial_params[key] = val

        # Save spectrum + parameters
        #run_id = db.add_run(eigvals, eigvecs, serial_params)
        run_id = db.add_run(eigvals, eigvecs,params)
        print(f"Saved run {run_id}: N={N}, method={method}, nev={len(eigvals)}")




