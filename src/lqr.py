from scipy.linalg import solve_discrete_are
from numpy.linalg import solve
import numpy as np
import warnings


def lqr_lti(A, B, Q, R):
    # P = solve_discrete_are(A, B, Q, R)
    Q_adjusted = 0.5 * Q
    R_adjusted = 0.5 * R
    P = np.ones_like(A)
    K = solve(R_adjusted + B.T @ P @ B, -B.T @ P @ A)
    while True:
        P = Q_adjusted + K.T @ R_adjusted @ K + (A + B @ K).T @ P @ (A + B @ K)
        Knew = solve(R_adjusted + B.T @ P @ B, -B.T @ P @ A)
        if np.linalg.norm(Knew - K) < 1e-6:
            break
        K = Knew.copy()
    return K, P


def lqr_ltv(A, B, Q, R, Qfinal):
    Q_adjusted = 0.5 * Q
    R_adjusted = 0.5 * R
    Qfinal_adjusted = 0.5 * Qfinal
    H = len(A)
    P = [np.zeros_like(A[0]) for k in range(H + 1)]
    K = [np.zeros_like(B[0]).T for k in range(H)]

    P[H] = Qfinal_adjusted.copy()
    for t in range(H - 1, -1, -1):
        K[t] = solve(R_adjusted + B[t].T @ P[t + 1] @ B[t], -B[t].T @ P[t + 1] @ A[t])
        P[t] = (
            Q_adjusted
            + K[t].T @ R_adjusted @ K[t]
            + (A[t] + B[t] @ K[t]).T @ P[t + 1] @ (A[t] + B[t] @ K[t])
        )

    return K, P


def lqr_linearized_tv(A, B, C_x, C_u, C_xx, C_uu, C_x_f, C_xx_f):
    H = len(A)

    k = [np.zeros(B[0].shape[1]) for _ in range(H)]
    K = [np.zeros_like(B[0]).T for _ in range(H)]
    V_x = C_x_f.copy()
    V_xx = C_xx_f.copy()

    for t in range(H - 1, -1, -1):
        A_t, B_t = A[t], B[t]
        C_x_t, C_u_t, C_xx_t, C_uu_t = C_x[t], C_u[t], C_xx[t], C_uu[t]

        Q_x = C_x_t + A_t.T @ V_x
        Q_u = C_u_t + B_t.T @ V_x

        Q_xx = C_xx_t + A_t.T @ V_xx @ A_t
        Q_ux = B_t.T @ V_xx @ A_t
        Q_uu = C_uu_t + B_t.T @ V_xx @ B_t

        K[t] = solve(-Q_uu, Q_ux)
        k[t] = solve(-Q_uu, Q_u)

        V_x = Q_x - K[t].T @ Q_uu @ k[t]
        V_xx = Q_xx - K[t].T @ Q_uu @ K[t]
        V_xx = (V_xx + V_xx.T) / 2.0

    return k, K
