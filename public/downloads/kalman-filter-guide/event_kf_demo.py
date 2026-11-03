# -*- coding: utf-8 -*-
"""教学示例：模拟的是事件前端输出的特征位置，不是原始事件。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def predict(x, P, dt, q):
    if dt < 0:
        raise ValueError('观测时间戳必须递增；乱序数据应先排序或回放')
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1.]])
    Q = q*np.array([[dt**3/3,0,dt**2/2,0],
                    [0,dt**3/3,0,dt**2/2],
                    [dt**2/2,0,dt,0],[0,dt**2/2,0,dt]])
    return F@x, F@P@F.T+Q

def update(x, P, z, R, gate=9.21034):
    H = np.array([[1.,0,0,0],[0,1.,0,0]])
    innovation = z-H@x
    S = H@P@H.T+R
    nis = float(innovation@np.linalg.solve(S, innovation))
    if nis > gate:
        return x, P, nis, False
    K = np.linalg.solve(S, H@P).T
    x = x+K@innovation
    A = np.eye(4)-K@H
    P = A@P@A.T+K@R@K.T
    return x, (P+P.T)/2, nis, True

def main():
    rng = np.random.default_rng(42)
    t = np.r_[0., np.cumsum(rng.uniform(.005,.025,320))]
    truth = np.column_stack([60+42*t,80+18*t])
    sigma = 1.5
    z = truth+rng.normal(0,sigma,truth.shape)
    bad = np.arange(37,len(t),43)
    z[bad] += [28,-22]
    # 第一条观测仅用于初始化，不再重复更新。
    x = np.array([z[0,0],z[0,1],0.,0.])
    P = np.diag([sigma**2,sigma**2,10000.,10000.])
    R = np.eye(2)*sigma**2
    estimates=[x.copy()]; covariances=[P.copy()]; ns=[np.nan]; accepted=[True]
    for k in range(1,len(t)):
        x,P=predict(x,P,t[k]-t[k-1],q=10.)
        x,P,nis,ok=update(x,P,z[k],R)
        assert np.linalg.eigvalsh(P).min()>-1e-10
        estimates.append(x.copy());covariances.append(P.copy());ns.append(nis);accepted.append(ok)
    estimates=np.array(estimates); covariances=np.array(covariances); accepted=np.array(accepted)
    warm=t>.5
    raw_rmse=np.sqrt(np.mean(np.sum((z[warm]-truth[warm])**2,axis=1)))
    kf_rmse=np.sqrt(np.mean(np.sum((estimates[warm,:2]-truth[warm])**2,axis=1)))
    print(f'2D position RMSE after 0.5 s: raw={raw_rmse:.3f} px, KF={kf_rmse:.3f} px')
    print(f'Rejected observations: {np.sum(~accepted)} / {len(t)-1}')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axs=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    axs[0].plot(*truth.T,label='真实轨迹',lw=2);axs[0].scatter(*z.T,s=9,alpha=.35,label='带噪特征位置');axs[0].plot(*estimates[:,:2].T,label='KF 估计');axs[0].scatter(*z[~accepted].T,marker='x',color='red',label='门控拒绝');axs[0].set(xlabel='u / px',ylabel='v / px',title='异步二维特征跟踪：合成数据');axs[0].legend()
    err=estimates[:,0]-truth[:,0];bound=2*np.sqrt(covariances[:,0,0]);axs[1].plot(t,err,label='u 位置误差');axs[1].fill_between(t,-bound,bound,alpha=.2,label='估计 ±2σ（非严格覆盖保证）');axs[1].set(xlabel='时间 / s',ylabel='误差 / px');axs[1].legend()
    out=Path(__file__).parent/'卡尔曼滤波学习指南_配图';out.mkdir(exist_ok=True)
    fig.savefig(out/'07_异步跟踪实验.png',dpi=170);plt.close(fig)
    (out/'实验结果.txt').write_text(f'固定随机种子 42；合成异步特征观测；启动后 0.5 s 统计。\n二维位置 RMSE：原始 {raw_rmse:.3f} px；KF {kf_rmse:.3f} px。\n拒绝 {np.sum(~accepted)} / {len(t)-1} 条观测。\n',encoding='utf-8')

if __name__=='__main__': main()
