# -*- coding: utf-8 -*-
"""卡尔曼滤波教学代码。数组均为 NumPy 浮点数组，状态为一维向量。
欧氏算法可直接调用；ESKF、InEKF、MSCKF 提供核心算子，不是完整 VIO。
"""
import numpy as np

# %% 公共运算、标准 KF 与异步运动模型

def sym(P):
    """消除浮点运算产生的微小非对称，不修复错误模型。"""
    return (P + P.T) / 2


def correct(x, P, residual, H, R, gate=np.inf):
    """由残差更新状态；返回状态、协方差、NIS 和是否接受观测。"""
    S = sym(H @ P @ H.T + R)
    nis = float(residual @ np.linalg.solve(S, residual))
    if nis > gate:
        return x.copy(), P.copy(), nis, False
    # 解线性方程求增益，避免显式求逆。
    K = np.linalg.solve(S, H @ P).T
    A = np.eye(len(x)) - K @ H
    return x + K @ residual, sym(A @ P @ A.T + K @ R @ K.T), nis, True


def kf_predict(x, P, F, Q, control=None):
    """control 是已计算好的 B @ u，没有控制时传 None。"""
    return F @ x + (0 if control is None else control), sym(F @ P @ F.T + Q)


def kf_update(x, P, z, H, R, gate=np.inf):
    return correct(x, P, z - H @ x, H, R, gate)


def cv_model(dt, q):
    """二维匀速模型；dt 单位秒，q 为连续白加速度谱密度。"""
    if dt < 0 or q < 0:
        raise ValueError("时间差和谱密度不能为负")
    F = np.eye(4)
    F[:2, 2:] = dt * np.eye(2)
    Q = q * np.block([[dt**3/3*np.eye(2), dt**2/2*np.eye(2)],
                      [dt**2/2*np.eye(2), dt*np.eye(2)]])
    return F, Q

# %% EKF 与迭代 EKF

def ekf_predict(x, P, f, jac_f, Q):
    """加性噪声模型；输入和 dt 可由调用者封装在 f、jac_f 中。"""
    F = jac_f(x)
    return f(x), sym(F @ P @ F.T + Q)


def ekf_update(x, P, z, h, jac_h, R):
    """只处理欧氏残差；角度观测需要另外实现角度归一化。"""
    return correct(x, P, z - h(x), jac_h(x), R)


def iterated_ekf(x, P, z, h, jac_h, R, max_iter=10, tol=1e-8):
    """固定同一个先验，反复重线性化；最后仅更新一次协方差。"""
    if max_iter < 1:
        raise ValueError("max_iter 至少为 1")
    xi = x.copy()
    for _ in range(max_iter):
        H = jac_h(xi)
        K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
        xn = x + K @ (z - h(xi) + H @ (xi - x))
        if np.linalg.norm(xn - xi) < tol:
            xi = xn
            break
        xi = xn
    H = jac_h(xi)
    # 最终线性化的 Joseph 更新；不把每次迭代看成新测量。
    _, Pn, _, _ = correct(x, P, z - h(xi), H, R)
    return xi, Pn

# %% UKF 与 CKF

def sigma_points(x, P, kind="ukf", alpha=1., beta=2., kappa=0.):
    """P 须正定。返回代表点、均值权重和协方差权重。"""
    n = len(x)
    L = np.linalg.cholesky(P)
    if kind == "ckf":
        offsets = np.sqrt(n) * L.T
        w = np.full(2*n, 1/(2*n))
        return np.vstack((x + offsets, x - offsets)), w, w.copy()
    if kind != "ukf":
        raise ValueError("kind 只能是 ukf 或 ckf")
    c = alpha**2 * (n + kappa)
    if c <= 0:
        raise ValueError("sigma 点尺度必须为正")
    offsets = np.sqrt(c) * L.T
    wm = np.full(2*n+1, 1/(2*c))
    wm[0] = (c-n)/c
    wc = wm.copy()
    wc[0] += 1-alpha**2+beta
    return np.vstack((x, x+offsets, x-offsets)), wm, wc


def moment(points, wm, wc, noise):
    mean = wm @ points
    delta = points - mean
    return mean, sym((delta.T * wc) @ delta + noise)


def sigma_step(x, P, z, f, h, Q, R, kind="ukf"):
    """UKF/CKF 的完整一步，限欧氏状态、加性噪声。"""
    points, wm, wc = sigma_points(x, P, kind)
    xp, Pp = moment(np.array([f(p) for p in points]), wm, wc, Q)
    # 加入 Q 后重新生成点，使测量传播包含过程噪声。
    points, wm, wc = sigma_points(xp, Pp, kind)
    obs = np.array([h(p) for p in points])
    zp, S = moment(obs, wm, wc, R)
    cross = ((points-xp).T * wc) @ (obs-zp)
    K = np.linalg.solve(S, cross.T).T
    return xp + K @ (z-zp), sym(Pp-K @ S @ K.T)

# %% ESKF：右乘局部姿态误差的核心算子

def skew(v):
    a, b, c = v
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def so3_exp(v):
    """旋转向量转旋转矩阵；小角度使用级数避免除零。"""
    t = np.linalg.norm(v)
    W = skew(v)
    a = 1-t*t/6 if t < 1e-6 else np.sin(t)/t
    b = .5-t*t/24 if t < 1e-6 else (1-np.cos(t))/t**2
    return np.eye(3) + a*W + b*W@W


def eskf_predict(state, P, am, wm, gravity, Qc, dt):
    """state=(p,v,Rwb,ba,bg)，误差顺序 p,v,theta,ba,bg。
    Qc 为 12x12 连续噪声谱密度，顺序 na,ng,nba,nbg。
    使用短时间一阶协方差离散化；高动态场景需更准确积分。
    """
    if dt < 0:
        raise ValueError("dt 不能为负")
    p, v, R, ba, bg = state
    a, w = am-ba, wm-bg
    aw = R @ a + gravity
    nominal = (p+v*dt+.5*aw*dt**2, v+aw*dt, R@so3_exp(w*dt), ba.copy(), bg.copy())
    A = np.zeros((15, 15))
    A[:3, 3:6] = np.eye(3)
    A[3:6, 6:9], A[3:6, 9:12] = -R@skew(a), -R
    A[6:9, 6:9], A[6:9, 12:15] = -skew(w), -np.eye(3)
    G = np.zeros((15, 12))
    G[3:6, :3], G[6:9, 3:6] = -R, -np.eye(3)
    G[9:12, 6:9], G[12:15, 9:12] = np.eye(3), np.eye(3)
    Phi = np.eye(15) + A*dt
    return nominal, sym(Phi@P@Phi.T + G@Qc@G.T*dt)


def eskf_correct(state, P, residual, H_error, R_obs):
    """H_error 必须相对于本文右乘误差推导，不是四元数四分量雅可比。"""
    dx, Pn, _, _ = correct(np.zeros(15), P, residual, H_error, R_obs)
    p, v, R, ba, bg = state
    state = (p+dx[:3], v+dx[3:6], R@so3_exp(dx[6:9]), ba+dx[9:12], bg+dx[12:15])
    J = np.eye(15)
    J[6:9, 6:9] -= .5*skew(dx[6:9])  # 注入后，协方差也须重置。
    return state, sym(J@Pn@J.T)

# %% InEKF：由具体群模型提供误差传播与注入

def inekf_core(X, P, residual, H, R, inject_and_reset):
    """不是通用导航器：调用者必须提供与群误差定义一致的残差、H 和注入。
    回调接收 (X, dx)，返回 (更新后的群状态, 误差重置雅可比)。
    对正文 eta= Xhat @ inv(Xtrue)，注入为 Exp(-hat(dx)) @ Xhat。
    偏置、外参与观测的处理取决于具体群模型，不能照搬 ESKF 的 J。
    """
    dx, Pn, _, _ = correct(np.zeros(P.shape[0]), P, residual, H, R)
    Xn, J = inject_and_reset(X, dx)
    return Xn, sym(J @ Pn @ J.T)

# %% MSCKF：白化后消除地标误差

def msckf_project(residual, Hx, Hf, R):
    """输入堆叠后的多视角线性残差；返回消元后的残差、状态雅可比和单位噪声。
    前端关联、位姿克隆、三角化和可观性修正仍需由完整系统实现。
    """
    L = np.linalg.cholesky(R)
    r = np.linalg.solve(L, residual)
    A = np.linalg.solve(L, Hx)
    B = np.linalg.solve(L, Hf)
    U, s, _ = np.linalg.svd(B, full_matrices=True)
    threshold = np.finfo(float).eps * max(B.shape) * (s[0] if len(s) else 0)
    rank = np.count_nonzero(s > threshold)
    N = U[:, rank:]  # N.T @ B 约为零，地标误差被消去。
    return N.T@r, N.T@A, np.eye(N.shape[1])

# %% 自适应测量噪声与鲁棒更新

def adaptive_R(R, residual, H, P_prior, rate=.02, floor=1e-6):
    """协方差匹配教学近似：用当前创新估计下一次更新的 R。
    输入必须是门控前统计；实际应使用窗口并处理离群点和相关性。
    """
    if not 0 < rate <= 1 or floor <= 0:
        raise ValueError("rate 需在 (0,1] 内，floor 为正")
    estimate = sym((1-rate)*R + rate*(np.outer(residual,residual)-H@P_prior@H.T))
    vals, vecs = np.linalg.eigh(estimate)
    return sym((vecs*np.maximum(vals, floor))@vecs.T)


def robust_update(x, P, z, H, R, huber=2.5, gate=np.inf):
    """先用原始创新门控，再用一个整体 Huber 权重放大 R；工程近似。"""
    r = z-H@x
    S = H@P@H.T+R
    d2 = float(r@np.linalg.solve(S,r))
    if d2 > gate:
        return x.copy(), P.copy(), d2, False
    weight = min(1., huber/max(np.sqrt(d2), 1e-12))
    return correct(x, P, r, H, R/weight)

# %% 平方根 KF 与正权重的平方根 UKF

def qr_factor(columns):
    """返回 L，使 L@L.T = columns@columns.T；统一三角因子符号。"""
    _, T = np.linalg.qr(columns.T, mode="reduced")
    signs = np.where(np.diag(T) < 0, -1., 1.)
    return T.T * signs


def sr_kf_step(x, L, z, F, H, Lq, Lr):
    """L、Lq、Lr 分别是 P、Q、R 的平方根。更新用 Joseph 因子的 QR。"""
    xp = F@x
    Lp = qr_factor(np.hstack((F@L,Lq)))
    S = H@Lp@Lp.T@H.T + Lr@Lr.T
    K = np.linalg.solve(S, H@Lp@Lp.T).T
    Ln = qr_factor(np.hstack(((np.eye(len(x))-K@H)@Lp, K@Lr)))
    return xp+K@(z-H@xp), Ln


def sr_ukf_step(x, L, z, f, h, Lq, Lr):
    """固定 alpha=1,beta=2,kappa=0，使协方差权重非负。
    用 QR 构造预测、创新和更新因子；不支持负权重参数。
    """
    pts, wm, wc = sigma_points(x, L@L.T)
    propagated = np.array([f(p) for p in pts])
    xp = wm@propagated
    Lp = qr_factor(np.hstack(((propagated-xp).T*np.sqrt(wc),Lq)))
    pts, wm, wc = sigma_points(xp,Lp@Lp.T)
    obs = np.array([h(p) for p in pts])
    zp = wm@obs
    X, Z = (pts-xp).T, (obs-zp).T
    Ls = qr_factor(np.hstack((Z*np.sqrt(wc),Lr)))
    cross = (X*wc)@Z.T
    K = np.linalg.solve(Ls.T,np.linalg.solve(Ls,cross.T)).T
    Ln = qr_factor(np.hstack(((X-K@Z)*np.sqrt(wc),K@Lr)))
    return xp+K@(z-zp), Ln

# %% IMM：相同状态维度、不同线性运动模型

def imm_step(xs, Ps, mu, transition, models, z, H, R):
    """transition[i,j]=从模型 i 转到 j 的概率，models 是 (F,Q) 列表。
    要求所有模型共用状态坐标；每个目标模型的预测概率必须为正。
    """
    prior = mu@transition
    if np.any(prior <= 0):
        raise ValueError("各模型预测概率须为正")
    new_x, new_P, log_likelihood = [], [], []
    for j, (F,Q) in enumerate(models):
        weights = mu*transition[:,j]/prior[j]
        xm = weights@xs
        Pm = sum(w*(P+np.outer(x-xm,x-xm)) for w,x,P in zip(weights,xs,Ps))
        xp,Pp = kf_predict(xm,Pm,F,Q)
        r,S = z-H@xp, H@Pp@H.T+R
        # 对数似然，避免连续更新后概率下溢。
        log_likelihood.append(-.5*(len(z)*np.log(2*np.pi)+np.linalg.slogdet(S)[1]+r@np.linalg.solve(S,r)))
        xn,Pn,_,_ = kf_update(xp,Pp,z,H,R)
        new_x.append(xn); new_P.append(Pn)
    logw = np.log(prior)+log_likelihood
    weights = np.exp(logw-np.max(logw)); weights /= weights.sum()
    xs,Ps = np.array(new_x),np.array(new_P)
    mean = weights@xs
    cov = sum(w*(P+np.outer(x-mean,x-mean)) for w,x,P in zip(weights,xs,Ps))
    return xs,Ps,weights,mean,sym(cov)

# %% 信息滤波与 RTS 平滑

def information_update(info, eta, z, H, R):
    """info=P^-1，eta=info@x；仅融合与已有信息独立的观测。"""
    return sym(info+H.T@np.linalg.solve(R,H)), eta+H.T@np.linalg.solve(R,z)


def information_predict(info, eta, F, Q):
    """教学实现先恢复均值与协方差再传播，未利用稀疏结构。"""
    P = np.linalg.solve(info,np.eye(len(eta)))
    xp,Pp = kf_predict(P@eta,P,F,Q)
    out = np.linalg.solve(Pp,np.eye(len(eta)))
    return sym(out),out@xp


def rts_smooth(filtered_x, filtered_P, predicted_x, predicted_P, F):
    """F[k] 从 k 传播到 k+1；predicted_*[k+1] 是该时刻更新前预测。
    输入按时间正序保存，输出使用未来观测修正后的欧氏状态序列。
    """
    xs,Ps = np.array(filtered_x,copy=True),np.array(filtered_P,copy=True)
    for k in range(len(xs)-2,-1,-1):
        G = np.linalg.solve(predicted_P[k+1], F[k]@filtered_P[k]).T
        xs[k] = filtered_x[k]+G@(xs[k+1]-predicted_x[k+1])
        Ps[k] = sym(filtered_P[k]+G@(Ps[k+1]-predicted_P[k+1])@G.T)
    return xs,Ps

# %% EnKF 与 bootstrap 粒子滤波

def enkf_step(ensemble, z, f, h, Q, R, rng):
    """每行一个状态样本；使用扰动观测版本，至少需要两个样本。"""
    count,n = ensemble.shape
    if count < 2:
        raise ValueError("集合至少包含两个样本")
    X = np.array([f(x) for x in ensemble])+rng.multivariate_normal(np.zeros(n),Q,count)
    Z = np.array([h(x) for x in X])
    A,B = X-X.mean(axis=0),Z-Z.mean(axis=0)
    K = np.linalg.solve(B.T@B/(count-1)+R, B.T@A/(count-1)).T
    noise = rng.multivariate_normal(np.zeros(len(z)),R,count)
    return X+(z+noise-Z)@K.T


def particle_step(particles, weights, z, f, h, Q, R, rng):
    """bootstrap PF，欧氏加性高斯模型；返回重采样后的等权粒子。
    这是贝叶斯方法对照，不属于卡尔曼滤波。此教学版每步重采样。
    """
    count,n = particles.shape
    X = np.array([f(x) for x in particles])+rng.multivariate_normal(np.zeros(n),Q,count)
    residuals = z-np.array([h(x) for x in X])
    with np.errstate(divide="ignore"):
        logw = np.log(weights)-.5*np.sum(residuals*np.linalg.solve(R,residuals.T).T,axis=1)
    logw -= np.max(logw)
    w = np.exp(logw); w /= w.sum()
    # 系统重采样：一个随机起点加等间距采样，保留高权重粒子。
    positions = (rng.random()+np.arange(count))/count
    cdf = np.cumsum(w); cdf[-1] = 1.
    return X[np.searchsorted(cdf,positions)],np.full(count,1/count)

# %% 可直接运行的最小示例

def demo():
    """验证正文的一维手算，并比较线性模型下的几种实现。"""
    x,P = np.array([2.]),np.array([[4.]])
    z,H,R = np.array([5.]),np.array([[1.]]),np.array([[1.]])
    xn,Pn,_,_ = kf_update(x,P,z,H,R)
    print("KF 后验均值、方差：",xn[0],Pn[0,0])
    np.testing.assert_allclose(xn,[4.4])
    np.testing.assert_allclose(Pn,[[.8]])
    f,h = lambda v:v,lambda v:v
    for name in ("ukf","ckf"):
        a,b = sigma_step(x,P,z,f,h,np.zeros((1,1)),R,name)
        np.testing.assert_allclose(a,xn)
        np.testing.assert_allclose(b,Pn)
        print(name.upper(),a[0],b[0,0])
    print("最小示例通过。更多方法需按各函数说明提供模型和输入。")


if __name__ == "__main__":
    demo()