from __future__ import print_function

import numpy as np
from lap import lapjv

from .kalman import KalmanTracker, TrackStatus


# ----------------- small utilities (cache + safe lower-bound pruning) -----------------

def _eig2x2_minmax(M):
    """返回 2x2 对称矩阵的 (lambda_min, lambda_max)，假设近似对称。"""
    a = M[..., 0, 0]
    b = 0.5 * (M[..., 0, 1] + M[..., 1, 0])  # 对称化
    d = M[..., 1, 1]
    tr = a + d
    det = a * d - b * b
    disc = np.clip(tr * tr - 4.0 * det, 0.0, None)
    s = np.sqrt(disc)
    lam_max = 0.5 * (tr + s)
    lam_min = 0.5 * (tr - s)
    return lam_min, lam_max


class _FrameCostCache(object):
    """
    帧内缓存：
      - (det_idx, trk_idx) -> exact distance（float）
      - det 侧: y、R 及 R 的特征值上下界
      - trk 侧: z_pred、Q_t 及 Q_t 的特征值上下界
    """
    __slots__ = ("cost", "y_all", "R_all", "R_lmin", "R_lmax",
                 "z_all", "Qt_all", "Qt_lmin", "Qt_lmax")

    def __init__(self, dets):
        self.cost = {}
        # det 侧缓存
        self.y_all = [d.y for d in dets]   # 2x1
        self.R_all = [d.R for d in dets]   # 2x2
        if len(dets) > 0:
            R_arr = np.stack(self.R_all, axis=0).astype(np.float64)  # (D,2,2)
            R_lmin, R_lmax = _eig2x2_minmax(R_arr)
            self.R_lmin = R_lmin
            self.R_lmax = R_lmax
        else:
            self.R_lmin = np.empty((0,), dtype=np.float64)
            self.R_lmax = np.empty((0,), dtype=np.float64)

        # trk 侧缓存占位，稍后由 prepare_trackers() 填充
        self.z_all  = None
        self.Qt_all = None
        self.Qt_lmin = None
        self.Qt_lmax = None

    def prepare_trackers(self, trackers):
        if len(trackers) == 0:
            self.z_all  = np.empty((0, 2), dtype=np.float64)
            self.Qt_all = np.empty((0, 2, 2), dtype=np.float64)
            self.Qt_lmin = np.empty((0,), dtype=np.float64)
            self.Qt_lmax = np.empty((0,), dtype=np.float64)
            return
        self.z_all  = np.stack([t.z_pred.reshape(2) for t in trackers], axis=0).astype(np.float64)   # (T,2)
        self.Qt_all = np.stack([t.Q_t for t in trackers], axis=0).astype(np.float64)                 # (T,2,2)
        Qt_lmin, Qt_lmax = _eig2x2_minmax(self.Qt_all)
        self.Qt_lmin = Qt_lmin
        self.Qt_lmax = Qt_lmax


def _build_cost_matrix_cached_pruned(det_indices, trk_indices, cache, trackers, thresh):
    """
    数值等价保障：
      - 最终进入 lapjv 的 cost(i,j) 仍来自 trackers[t].distance(y,R) 的精确值；
      - 仅当 “严格下界 > thresh” 时直接赋值为 (thresh + 1.0)，等价于被 cost_limit 忽略；
      - 这样不会改变 lapjv 的最优匹配结果，只是减少了 distance 的调用次数。
    仍保持 float64。
    """
    D, T = len(det_indices), len(trk_indices)
    if D == 0 or T == 0:
        return np.zeros((D, T))  # float64

    C = np.empty((D, T))         # float64

    # 选取本轮使用到的 track 索引对应的缓存切片
    z_sel      = cache.z_all[trk_indices]           # (T,2)
    Qt_lmin_s  = cache.Qt_lmin[trk_indices]         # (T,)
    Qt_lmax_s  = cache.Qt_lmax[trk_indices]         # (T,)

    tiny = 1e-18

    for i, di in enumerate(det_indices):
        y = cache.y_all[di].reshape(2)              # (2,)
        R = cache.R_all[di]
        R_lmin = cache.R_lmin[di]
        R_lmax = cache.R_lmax[di]

        # pair 下界：d >= ||y-z||^2 / (λmax_Qt + λmax_R) + 2 * log(λmin_Qt + λmin_R)
        diff2 = np.sum((z_sel - y[None, :])**2, axis=1)   # (T,)
        lam_max_S = Qt_lmax_s + R_lmax
        lam_min_S = Qt_lmin_s + R_lmin
        lam_min_S = np.maximum(lam_min_S, tiny)

        lower_bound = diff2 / lam_max_S + 2.0 * np.log(lam_min_S)

        row = C[i]
        # 需要精确计算的对（下界不超过阈值）
        need_exact = lower_bound <= thresh
        # 直接可剪的对（下界已超阈值）
        row[~need_exact] = thresh + 1.0

        # 精确计算：命中过缓存则复用，否则调用 distance
        idxs = np.nonzero(need_exact)[0]
        for k in idxs:
            tj = trk_indices[k]
            key = (di, tj)
            v = cache.cost.get(key)
            if v is None:
                v = float(trackers[tj].distance(cache.y_all[di], R))
                cache.cost[key] = v
            row[k] = v

    return C


def linear_assignment(cost_matrix, thresh):
    """保持与原版一致的匹配实现（float64 + cost_limit）。"""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = tuple([i for i, j in enumerate(x) if j < 0])
    unmatched_b = tuple([i for i, j in enumerate(y) if j < 0])
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


# --------------- tracker core ---------------

class UCMCTrack(object):
    def __init__(self, a1, a2, wx, wy, vmax, max_age, fps, dataset, high_score, use_cmc, detector=None):
        self.wx = wx
        self.wy = wy
        self.vmax = vmax
        self.dataset = dataset
        self.high_score = high_score
        self.max_age = max_age
        self.a1 = a1
        self.a2 = a2
        self.dt = 1.0 / float(fps) if fps else 1.0 / 30.0

        self.use_cmc = use_cmc
        self.detector = detector

        self.trackers = []
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []

        self.detidx_remain = []  # used between rounds

    # ---------- per-frame bookkeeping ----------
    def _rebuild_index_lists(self, dets):
        self.confirmed_idx, self.coasted_idx, self.tentative_idx = [], [], []
        for i, t in enumerate(self.trackers):
            detidx = getattr(t, "detidx", -1)
            if 0 <= detidx < len(dets):
                if hasattr(dets[detidx], "bb_height"):
                    t.h = dets[detidx].bb_height
                if hasattr(dets[detidx], "bb_width"):
                    t.w = dets[detidx].bb_width
            if t.status == TrackStatus.Confirmed:
                self.confirmed_idx.append(i)
            elif t.status == TrackStatus.Coasted:
                self.coasted_idx.append(i)
            else:
                self.tentative_idx.append(i)

    # ---------- tracker lifecycle helpers ----------
    def _start_new_track(self, det):
        trk = KalmanTracker(det.y, det.R, self.wx, self.wy, self.vmax,
                            getattr(det, "bb_width", 0.0), getattr(det, "bb_height", 0.0), self.dt)
        trk.status = TrackStatus.Tentative
        trk.detidx = -1
        self.trackers.append(trk)

    def _purge_deleted(self):
        self.trackers = [t for t in self.trackers if t.status != TrackStatus.Coasted or t.death_count <= self.max_age]

    # ---------- association pipeline ----------
    def update(self, dets, frame_id):
        # 帧内缓存（含 det 侧统计）
        _cache = _FrameCostCache(dets)

        # 1) split by score
        detidx_high, detidx_low = [], []
        for i, d in enumerate(dets):
            if getattr(d, "conf", 0.0) >= self.high_score:
                detidx_high.append(i)
            else:
                detidx_low.append(i)

        # 2) predict & optional CMC
        for track in self.trackers:
            track.predict()
            if self.use_cmc and self.detector is not None:
                x, y = self.detector.cmc(track.kf.x[0, 0], track.kf.x[2, 0], track.w, track.h, frame_id)
                track.kf.x[0, 0] = x
                track.kf.x[2, 0] = y
                track.recache_after_state_change()

        # 准备 trk 侧缓存（需在 predict/CMC 之后）
        _cache.prepare_trackers(self.trackers)

        # build group indices before association
        self._rebuild_index_lists(dets)

        # 3) 高置信匹配（confirmed+coasted vs high）—— 使用 a1；不把 unmatched 置为 Coasted
        trackidx = self.confirmed_idx + self.coasted_idx
        for trk in self.trackers:
            trk.detidx = -1

        trackidx_remain = []
        self.detidx_remain = []

        num_det = len(detidx_high)
        num_trk = len(trackidx)
        if num_det * num_trk > 0:
            cost_matrix = _build_cost_matrix_cached_pruned(detidx_high, trackidx, _cache, self.trackers, self.a1)
            matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, self.a1)

            # 第一轮 unmatched → 进入第二轮
            for i in unmatched_a:
                self.detidx_remain.append(detidx_high[i])
            for i in unmatched_b:
                trackidx_remain.append(trackidx[i])

            for i, j in matched_indices:
                det_idx = detidx_high[i]
                trk_idx = trackidx[j]
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R)
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].birth_count += 1
                self.trackers[trk_idx].detidx = det_idx
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id
        else:
            self.detidx_remain = detidx_high[:]
            trackidx_remain = trackidx[:]

        # 4) 低置信匹配（remain tracks vs low）—— 使用 a2；此轮 unmatched 才置 Coasted
        num_det = len(detidx_low)
        num_trk = len(trackidx_remain)
        if num_det * num_trk > 0:
            cost_matrix = _build_cost_matrix_cached_pruned(detidx_low, trackidx_remain, _cache, self.trackers, self.a2)
            matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, self.a2)

            for i in unmatched_b:
                trk_idx = trackidx_remain[i]
                self.trackers[trk_idx].status = TrackStatus.Coasted
                self.trackers[trk_idx].detidx = -1

            for i, j in matched_indices:
                det_idx = detidx_low[i]
                trk_idx = trackidx_remain[j]
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R)
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].detidx = det_idx
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id

            self.detidx_remain.extend([detidx_low[i] for i in unmatched_a])
        else:
            self.detidx_remain.extend(detidx_low)

        # 5) Tentative 匹配（stricter a1）—— unmatched 仅清 detidx
        num_det = len(self.detidx_remain)
        num_trk = len(self.tentative_idx)
        if num_det * num_trk > 0:
            cost_matrix = _build_cost_matrix_cached_pruned(self.detidx_remain, self.tentative_idx, _cache, self.trackers, self.a1)
            matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, self.a1)

            for i, j in matched_indices:
                det_idx = self.detidx_remain[i]
                trk_idx = self.tentative_idx[j]
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R)
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].birth_count += 1
                self.trackers[trk_idx].detidx = det_idx
                dets[det_idx].track_id = self.trackers[trk_idx].id
                if self.trackers[trk_idx].birth_count >= 2:
                    self.trackers[trk_idx].birth_count = 0
                    self.trackers[trk_idx].status = TrackStatus.Confirmed

            for i in unmatched_b:
                trk_idx = self.tentative_idx[i]
                self.trackers[trk_idx].detidx = -1

            # 更新 remain dets
            self.detidx_remain = [self.detidx_remain[i] for i in unmatched_a]

        # 6) 为仍未匹配的 det 新建轨迹
        for det_idx in self.detidx_remain:
            self._start_new_track(dets[det_idx])
            self.trackers[-1].detidx = det_idx
            dets[det_idx].track_id = self.trackers[-1].id

        # 7) aging + 删除（保持原版）
        for t in self.trackers:
            if t.detidx == -1:
                t.death_count += 1
                if t.death_count > self.max_age:
                    t.status = TrackStatus.Coasted
        self._purge_deleted()

        # 8) rebuild 索引
        self._rebuild_index_lists(dets)