import sqlite3
import json
import numpy as np

class SpectrumDatabase:
    def __init__(self, path="spectra.db"):
        self.conn = sqlite3.connect(path)
        self._create_table()

    def _create_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS runs (
          id               INTEGER PRIMARY KEY AUTOINCREMENT,
          eigvals          BLOB    NOT NULL,
          eigvals_shape    TEXT    NOT NULL,
          eigvals_dtype    TEXT    NOT NULL,
          eigvecs          BLOB    NOT NULL,
          eigvecs_shape    TEXT    NOT NULL,
          eigvecs_dtype    TEXT    NOT NULL,
          params           TEXT    NOT NULL,
          created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
        );"""
        self.conn.execute(sql)
        self.conn.commit()

    def add_run(self,
                eigvals: np.ndarray,
                eigvecs: np.ndarray,
                params: dict) -> int:
        # serialize eigenvalues
        ev_b    = eigvals.tobytes()
        ev_shape= json.dumps(eigvals.shape)
        ev_dtype= str(eigvals.dtype)
        # serialize eigenvectors
        vec_b    = eigvecs.tobytes()
        vec_shape= json.dumps(eigvecs.shape)
        vec_dtype= str(eigvecs.dtype)
        # serialize params
        pjson = json.dumps(params,sort_keys=True,separators=(",", ":"),default=lambda o: o.tolist())

        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO runs
               (eigvals, eigvals_shape, eigvals_dtype,
                eigvecs, eigvecs_shape, eigvecs_dtype,
                params)
             VALUES (?,?,?,?,?,?,?)""",
            (sqlite3.Binary(ev_b), ev_shape, ev_dtype,
             sqlite3.Binary(vec_b), vec_shape, vec_dtype,
             pjson)
        )
        self.conn.commit()
        return cur.lastrowid

    #takes run_id and returns the eigenspectrum
    def get_run_id(self, run_id: int):
        
        cur = self.conn.cursor()
        cur.execute(
            """SELECT eigvals, eigvals_shape, eigvals_dtype,
                      eigvecs, eigvecs_shape, eigvecs_dtype,
                      params
               FROM runs WHERE id = ?""",
            (run_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None

        ev_b, ev_shape, ev_dtype, vec_b, vec_shape, vec_dtype, params_txt = row
        # rebuild arrays
        ev_shape  = tuple(json.loads(ev_shape))
        ev = np.frombuffer(ev_b, dtype=ev_dtype).reshape(ev_shape)

        vec_shape = tuple(json.loads(vec_shape))
        vec = np.frombuffer(vec_b, dtype=vec_dtype).reshape(vec_shape)

        return ev, vec

    #takes the param and returns the id
    def get_run_param(self,params: dict):

        cur = self.conn.cursor()
        #convert the param dictionary to a sorted, spaceless, serializable JSON formatted strings
        canonical = json.dumps(params,sort_keys=True,separators=(",", ":"),default=lambda o: o.tolist())

        cur.execute("SELECT id FROM runs WHERE params = ?",(canonical,))
        ids_full_match = [row[0] for row in cur.fetchall()]

        return ids_full_match






