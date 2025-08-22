from filterpy.kalman import KalmanFilter
import numpy as np
from enum import Enum

class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted   = 2

class KalmanTracker(object):
    """
    保持接口不变；增加历史特征缓存 z_pred/Q_t（float64），用于快速/稳定的代价计算。
    """
    count = 1

    def __init__(self, y, R, wx, wy, vmax, w, h, dt=1/30):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        F = np.array([[1, dt, 0,  0],
                      [0,  1, 0,  0],
                      [0,  0, 1, dt],
                      [0,  0, 0,  1]], dtype=float)
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]], dtype=float)
        self.kf.F = F
        self.kf.H = H

        qx = max(float(wx), 1e-4)
        qy = max(float(wy), 1e-4)
        self.kf.Q = np.diag([qx, qx, qy, qy]) * 1e-2

        self.kf.R = R.copy().astype(float)
        self.kf.x = np.array([[y[0,0]], [0.0], [y[1,0]], [0.0]], dtype=float)
        self.kf.P = np.eye(4, dtype=float) * 1.0

        self.id = KalmanTracker.count
        KalmanTracker.count += 1

        self.status = TrackStatus.Tentative
        self.age = 0
        self.birth_count = 0
        self.death_count = 0
        self.detidx = -1
        self.time_since_update = 0

        self.w = float(w)
        self.h = float(h)

        # 历史特征缓存
        self.z_pred = None  # 2x1 = Hx
        self.Q_t    = None  # 2x2 = H P H^T
        self._cache_projected()

    def _cache_projected(self):
        # 用 float64 保存缓存，稳定且与向量化计算精度一致
        self.z_pred = (self.kf.H @ self.kf.x).astype(np.float64)
        self.Q_t    = (self.kf.H @ self.kf.P @ self.kf.H.T).astype(np.float64)

    def recache_after_state_change(self):
        self._cache_projected()

    def update(self, y, R):
        self.kf.R = R
        self.kf.update(y)
        self.time_since_update = 0
        self.death_count = 0
        self._cache_projected()

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self._cache_projected()
        return self.kf.H @ self.kf.x

    def get_state(self):
        return self.kf.x

    def distance(self, y, R):
        """
        与 baseline 完全一致：Mahalanobis + logdet（使用缓存）
        """
        y = y.astype(np.float64); R = R.astype(np.float64)
        diff = y - self.z_pred
        S = self.Q_t + R
        a, b, c, d = S[0,0], S[0,1], S[1,0], S[1,1]
        det = a*d - b*c
        if det <= 1e-12:
            SI = np.linalg.pinv(S)
            logdet = np.log(max(np.linalg.det(S), 1e-12))
        else:
            inv00 =  d/det; inv01 = -b/det
            inv10 = -c/det; inv11 =  a/det
            SI = np.array([[inv00, inv01],[inv10, inv11]], dtype=np.float64)
            logdet = np.log(det)
        mahal = float(diff.T @ (SI @ diff))
        return mahal + logdet