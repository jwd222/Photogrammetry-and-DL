| Flag                     | Use With                        | Min Points    | Number of Outputs            | Notes                                                                    |
| ------------------------ | ------------------------------- | ------------- | ---------------------------- | ------------------------------------------------------------------------ |
| `cv2.SOLVEPNP_P3P`       | `solvePnPGeneric` only          | **Exactly 4** | Multiple (up to 4 solutions) | Minimal solver for calibrated cameras. Returns multiple pose hypotheses. |
| `cv2.SOLVEPNP_AP3P`      | `solvePnPGeneric` only          | **Exactly 4** | Multiple (up to 4 solutions) | Alternative minimal solver, often more stable than P3P.                  |
| `cv2.SOLVEPNP_EPNP`      | `solvePnP` or `solvePnPGeneric` | ≥ 4           | Single (1 solution)          | Efficient PnP, fast but may be less accurate.                            |
| `cv2.SOLVEPNP_DLS`       | `solvePnPGeneric` only          | ≥ 6           | Multiple (can be multiple)   | Direct Least Squares method, accurate but slower.                        |
| `cv2.SOLVEPNP_ITERATIVE` | `solvePnP` or `solvePnPGeneric` | ≥ 4           | Single (1 solution)          | Classic iterative method, stable and accurate.                           |
| `cv2.SOLVEPNP_UPNP`      | `solvePnP` only                 | ≥ 4           | Single (1 solution)          | For uncalibrated cameras; less commonly used.                            |
| `cv2.SOLVEPNP_SQPNP`     | `solvePnP` only                 | ≥ 3           | Single (1 solution)          | Newer optimization-based method; very accurate.                          |
