"""Thí nghiệm kiểm chứng Exercises D2L 1–5; CPU, seed 42, dữ liệu thật.

Kết quả được cache theo mục để notebook không phải học lại khi chỉ đọc bài.
Chạy notebook với RECOMPUTE=True để tạo lại. Không dùng d2l để tránh các
pin phụ thuộc cũ; thuật toán vẫn là PyTorch và tham số gốc được ghi rõ.
"""
from contextlib import redirect_stdout
import cProfile
import inspect
import io
import json
import math
from pathlib import Path
import platform
import pstats
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from data import BASE, DATA, fashion, abalone

RESULTS = BASE / "results"
FASHION = None


def seed(value=42):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_num_threads(2)


class Record:
    def __init__(self, section):
        self.path = RESULTS / section.replace(".", "_")
        self.path.mkdir(parents=True, exist_ok=True)
        self.result = {"section": section, "tables": [], "figures": [], "notes": []}

    def table(self, name, rows):
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        path = self.path / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        self.result["tables"].append(str(path.relative_to(BASE)))
        return frame

    def figure(self, name):
        path = self.path / f"{name}.png"
        plt.gcf().tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()
        self.result["figures"].append(str(path.relative_to(BASE)))

    def note(self, message):
        self.result["notes"].append(str(message))


def experiment_2_1(r):
    X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    Y = torch.tensor([[2.,1,4,3],[1,2,3,4],[4,3,2,1]])
    r.table("tensor_operations", [{"operation": key, "result": str(value.tolist())} for key,value in {
        "X":X, "Y":Y, "X+Y":X+Y, "X-Y":X-Y, "X*Y":X*Y, "X/Y":X/Y,
        "X**Y":X**Y,"exp(X)":X.exp(),"X[-1]":X[-1],"X[1:3]":X[1:3],
        "X==Y":X==Y, "X<Y":X<Y, "X>Y":X>Y,
        "cat rows":torch.cat((X,Y),0), "cat columns":torch.cat((X,Y),1),
    }.items()])
    a,b = torch.arange(8).reshape(2,1,4), torch.arange(3).reshape(1,3,1)
    r.note(f"Broadcast {tuple(a.shape)} + {tuple(b.shape)} -> {tuple((a+b).shape)}: {(a+b).tolist()}")
    r.note(f"Index X[1,2]={X[1,2].item()}; sum={X.sum().item()}; NumPy round-trip shape={tuple(torch.from_numpy(X.numpy()).shape)}")
    edited=X.clone();edited[1,2]=99;edited[0:2,:]=12
    before=id(edited);edited+=Y
    r.note(f"Slice assignment then in-place +=: {edited.tolist()}; id preserved={before==id(edited)}")
    scalar=torch.tensor([3.5]);r.note(f"Tensor scalar conversions: item={scalar.item()}, float={float(scalar.item())}")


def experiment_2_2(r):
    df = abalone()
    r.table("abalone_head",df.head())
    r.table("abalone_schema",[{"column":c,"dtype":str(df[c].dtype),"missing_fraction":df[c].isna().mean()} for c in df])
    r.table("named_selection",df.loc[df.Sex=="F",["Sex","Length","Rings"]].head())
    rows=[]
    for copies in [1,10,100]:
        frame=pd.concat([df]*copies,ignore_index=True)
        path=r.path/"memory_sample.csv"
        frame.to_csv(path,index=False)
        t=time.perf_counter(); loaded=pd.read_csv(path); elapsed=time.perf_counter()-t
        rows.append({"rows":len(loaded),"file_bytes":path.stat().st_size,
                     "dataframe_bytes":int(loaded.memory_usage(deep=True).sum()),"read_seconds":elapsed})
    path.unlink()
    r.table("memory_measurements",rows)
    arr=df[["Length","Rings"]].to_numpy()
    np.save(r.path/"example.npy",arr)
    assert np.array_equal(arr,np.load(r.path/"example.npy"))
    image=Image.fromarray(np.arange(256,dtype=np.uint8).reshape(16,16))
    image.save(r.path/"example.png")
    assert Image.open(r.path/"example.png").size==(16,16)
    r.note(f"Abalone: {df.shape}, missing cells={df.isna().sum().sum()}; NumPy/Pillow round-trips verified.")


def benchmark_matmul(A,B,repeats=5):
    A@B
    times=[]
    for _ in range(repeats):
        t=time.perf_counter(); A@B; times.append(time.perf_counter()-t)
    return float(np.median(times))


def experiment_2_3(r):
    A=torch.arange(20.).reshape(5,4); B=A+1
    assert torch.equal(A.T.T,A)
    assert torch.equal((A+B).T,A.T+B.T)
    S=torch.randn(4,4); assert torch.equal((S+S.T).T,S+S.T)
    X=torch.arange(24.).reshape(2,3,4)
    r.table("shapes",[{"operation":f"sum dim {d}","shape":str(tuple(X.sum(d).shape))} for d in range(3)])
    try: A/A.sum(1)
    except RuntimeError as e: r.note("Expected broadcasting error: "+str(e).splitlines()[0])
    r.note(f"len(X)={len(X)}; norm={torch.linalg.norm(X).item():.6f}; flattened norm={torch.linalg.vector_norm(X.flatten()).item():.6f}")
    r.table("row_normalization",(A/A.sum(1,keepdim=True)).numpy())
    a,b,c=[torch.randn(100,200) for _ in range(3)]
    stacked=torch.stack([a,b,c],dim=2); assert torch.equal(stacked[:,:,1],b)
    r.note(f"Stack dim=2: {tuple(stacked.shape)}; recovered B verified")
    # Kích thước đề gốc được tính chính xác, đo bản thu nhỏ để tránh trung gian 4 GiB.
    r.table("original_chain_cost",[
        {"order":"(AB)C","scalar_multiplies":2**31+2**29,"intermediate_float32_bytes":2**15*4},
        {"order":"A(BC)","scalar_multiplies":2**35+2**40,"intermediate_float32_bytes":2**30*4},
    ])
    a,b,c=torch.randn(64,4096),torch.randn(4096,2),torch.randn(2,1024)
    rows=[]
    for name,fn in [("(AB)C",lambda:(a@b)@c),("A(BC)",lambda:a@(b@c))]:
        fn();t=time.perf_counter()
        for _ in range(5): fn()
        rows.append({"order":name,"seconds_per_call":(time.perf_counter()-t)/5,"scale":"1/16 each dimension"})
    r.table("chain_benchmark",rows)
    a,b=torch.randn(128,8192),torch.randn(8192,32); c=torch.randn(32,8192)
    r.table("layout_benchmark",[
        {"case":"AB contiguous","seconds":benchmark_matmul(a,b)},
        {"case":"AC.T noncontiguous","seconds":benchmark_matmul(a,c.T)},
        {"case":"C=B.T view, then A@C.T","seconds":benchmark_matmul(a,b.T.T)},
    ])


def experiment_2_4(r):
    x=np.linspace(.25,2,250)
    plt.figure(figsize=(7,4));plt.plot(x,x**3-1/x,label="x^3 - 1/x")
    plt.plot(x,4*(x-1),label="tangent at x=1");plt.scatter([1],[0]);plt.legend();plt.grid(alpha=.2)
    r.figure("function_and_tangent")
    x=torch.tensor([1.,2.],requires_grad=True);f=3*x[0]**2+5*x[1].exp()
    f.backward();r.note(f"Gradient at (1,2)={x.grad.tolist()}")


def experiment_2_5(r):
    x=torch.tensor(2.,requires_grad=True); y=x**2; y.backward()
    try: y.backward()
    except RuntimeError as e: r.note("Expected second backward error: "+str(e).splitlines()[0])
    x=torch.tensor(2.,requires_grad=True); y=x**2;y.backward(retain_graph=True);y.backward()
    r.note(f"Repeated backward with retain_graph accumulates grad: {x.grad.item()}")
    x=torch.linspace(-math.pi,math.pi,200,requires_grad=True);y=x.sin();y.sum().backward()
    plt.figure(figsize=(7,4));plt.plot(x.detach(),y.detach(),label="sin(x)");plt.plot(x.detach(),x.grad,label="autograd derivative");plt.legend();r.figure("sin_autograd")
    x=torch.tensor(1.3,dtype=torch.float64,requires_grad=True)
    f=torch.log(x*x)*torch.sin(x)+1/x
    auto=torch.autograd.grad(f,x)[0]
    analytic=2*torch.sin(x)/x+torch.log(x*x)*torch.cos(x)-1/(x*x)
    a=x*x; da=2*x; b=a.log(); db=da/a; c=x.sin(); dc=x.cos()
    forward=(db*c+b*dc)-1/(x*x)
    bar_b=c;bar_c=b;bar_a=bar_b/a
    reverse=bar_a*2*x+bar_c*x.cos()-1/(x*x)
    assert torch.allclose(auto,analytic)
    assert torch.allclose(auto,forward) and torch.allclose(auto,reverse)
    r.table("chain_rule",[{"x":x.item(),"autograd":auto.item(),"forward_chain_rule":forward.item(),"reverse_chain_rule":reverse.item()}])
    def dynamic(a):
        b=a*2
        while b.norm()<1000: b=b*2
        return b if b.sum()>0 else b*100
    a=torch.tensor([1.,2.,3.],requires_grad=True);out=dynamic(a);out.sum().backward()
    r.note(f"Vector output={out.tolist()}, VJP for sum={a.grad.tolist()}")
    r.note("Dependency graph: x -> x^2 -> log(x^2); x -> sin(x); multiply -> +1/x -> f.")


def experiment_2_6(r):
    p=.5;rows=[]
    for n in [10,100,1000,10000]:
        estimates=torch.distributions.Binomial(n,p).sample((3000,))/n
        rows.append({"n":n,"empirical_variance":estimates.var().item(),"theoretical_variance":p*(1-p)/n})
    df=r.table("coin_variance",rows)
    plt.figure(figsize=(6,4));plt.loglog(df.n,df.empirical_variance,"o-",label="empirical");plt.loglog(df.n,df.theoretical_variance,"--",label="p(1-p)/n");plt.legend();r.figure("variance_vs_n")
    prior=.0015
    r.table("bayes_exercise",[
        {"observations":"one positive","posterior":.99*prior/(.99*prior+.1*(1-prior))},
        {"observations":"two positives, correlated healthy tests","posterior":.99**2*prior/(.99**2*prior+.02*(1-prior))},
    ])
    r.table("joint_given_healthy",[{"D1":a,"D2":b,"probability":p} for a,b,p in [(0,0,.82),(0,1,.08),(1,0,.08),(1,1,.02)]])


def experiment_2_7(r):
    r.note("torch.add docstring:\n"+torch.add.__doc__[:1300])
    r.note("No numbered Exercises in section 2.7 of the supplied PDF.")


def synthetic(n=1000,d=2,noise=.01,seed_value=42,bias=4.2):
    g=torch.Generator().manual_seed(seed_value)
    X=torch.randn(n,d,generator=g)
    w=torch.tensor([2.,-3.4]) if d==2 else torch.ones(d)*.01
    y=X@w+bias+torch.randn(n,generator=g)*noise
    return X,y[:,None],w


def fit_regression(X,y,lr=.03,epochs=3,loss_kind="mse",init_std=.01,manual=False,decay=0.):
    seed()
    net=nn.Linear(X.shape[1],1)
    with torch.no_grad(): net.weight.normal_(0,init_std);net.bias.zero_()
    optimizer=torch.optim.SGD(net.parameters(),lr=lr,weight_decay=decay)
    losses={"mse":lambda p,t:((p-t)**2/2).mean(),"l1":nn.L1Loss(),"huber":nn.HuberLoss(delta=1.)}
    loader=DataLoader(TensorDataset(X,y),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42))
    trace=[]
    for epoch in range(epochs):
        for xb,yb in loader:
            if manual:
                p=xb@net.weight.T+net.bias
            else: p=net(xb)
            loss=losses[loss_kind](p,yb)
            optimizer.zero_grad();loss.backward()
            if manual:
                with torch.no_grad():
                    for parameter in net.parameters():
                        parameter.add_(parameter.grad,alpha=-lr)
            else: optimizer.step()
        with torch.no_grad(): trace.append(float(losses[loss_kind](net(X),y)))
    return net,trace


def experiment_3_1(r):
    x=torch.tensor([1.,2.,3.,100.])
    r.note(f"Squared-loss minimizer mean={x.mean().item()}, absolute-loss minimizers interval [2,3].")
    X,y,w=synthetic();A=torch.cat([X,torch.ones(len(X),1)],1)
    estimate=torch.linalg.lstsq(A,y).solution.flatten()
    r.table("analytic_regression",[{"parameter":name,"estimate":v.item()} for name,v in zip(["w1","w2","bias"],estimate)])


def experiment_3_2(r):
    class B:
        def __init__(self,a,b):
            pass
    b=B(1,2)
    r.note(f"Without assigning/saving hyperparameters: hasattr(a)={hasattr(b,'a')}; hasattr(b)={hasattr(b,'b')}")
    r.note("Full reference classes: https://d2l.ai/_modules/d2l/torch.html#HyperParameters; Module; DataModule; Trainer. See downloaded source_reference/d2l_torch.py.")


def experiment_3_3(r):
    X,y,_=synthetic(n=101)
    r.table("last_batch",[{"drop_last":drop,"batch_sizes":[len(x) for x,_ in DataLoader(TensorDataset(X,y),batch_size=32,drop_last=drop)]} for drop in [False,True]])
    class Fresh(IterableDataset):
        def __iter__(self):
            for _ in range(3):
                x=torch.randn(2);yield x,2*x[0]-3.4*x[1]+4.2
    class Fixed(IterableDataset):
        def __iter__(self):
            g=torch.Generator().manual_seed(42)
            for _ in range(3):
                x=torch.randn(2,generator=g);yield x,2*x[0]-3.4*x[1]+4.2
    fresh=Fresh();fixed=Fixed()
    r.note(f"Fresh equal on two iterations: {torch.equal(list(fresh)[0][0],list(fresh)[0][0])}; Fixed equal: {torch.equal(list(fixed)[0][0],list(fixed)[0][0])}")
    r.note(inspect.getsource(Fresh)+"\n"+inspect.getsource(Fixed))


def experiment_3_4(r):
    X,y,w=synthetic();rows=[]
    for std in [0.,.01,math.sqrt(.01),math.sqrt(1000)]:
        net,trace=fit_regression(X,y,init_std=std,manual=True)
        rows.append({"init_std":std,"init_variance":std**2,"loss":trace[-1],"w_error":float((net.weight.detach().flatten()-w).norm())})
    r.table("initialization",rows)
    rows=[]
    plt.figure(figsize=(7,4))
    for lr in [.003,.03,.3]:
        net,trace=fit_regression(X,y,lr=lr,epochs=10,manual=True)
        rows.append({"lr":lr,"final_loss":trace[-1]});plt.semilogy(range(1,11),trace,label=f"lr={lr}")
    plt.legend();r.figure("learning_rates");r.table("learning_rates",rows)
    # Ohm: V=R*I, xác định R bằng autograd.
    I=torch.linspace(.1,2,100);V=7*I
    R=torch.tensor(1.,requires_grad=True)
    for _ in range(200):
        loss=((R*I-V)**2).mean();loss.backward()
        with torch.no_grad(): R-=.1*R.grad;R.grad.zero_()
    r.note(f"Ohm fitted R={R.item():.6f}, true R=7")
    # Planck: dùng log B và log T để tránh tràn số, giữ T dương.
    wavelength=torch.linspace(400e-9,2500e-9,100,dtype=torch.float64)
    def log_planck(logT):
        h,c,k=6.62607015e-34,299792458.,1.380649e-23
        z=h*c/(wavelength*k*logT.exp())
        return math.log(2*h*c*c)-5*wavelength.log()-torch.log(torch.expm1(z))
    truth=log_planck(torch.tensor(math.log(4500.),dtype=torch.float64)).detach()
    logT=torch.tensor(math.log(3000.),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([logT],lr=.02)
    for _ in range(400):
        opt.zero_grad();loss=((log_planck(logT)-truth)**2).mean();loss.backward();opt.step()
    r.note(f"Planck fitted T={logT.exp().item():.3f} K, true 4500 K; synthetic noiseless spectrum")
    rows=[]
    for outlier in [False,True]:
        target=y.clone()
        if outlier: target[5]=10000
        for kind in ["mse","l1","huber"]:
            net,trace=fit_regression(X,target,epochs=10,loss_kind=kind,manual=True)
            rows.append({"outlier":outlier,"loss":kind,"w_error":float((net.weight.detach().flatten()-w).norm()),"bias_error":abs(net.bias.item()-4.2)})
    r.table("robust_losses",rows)
    r.note("Initialization: std=0.01 means variance=0.0001. Literal variance=0.01 in the exercise is tested separately with std=0.1; variance=1000 uses std=sqrt(1000).")


def experiment_3_5(r):
    rows=[]
    for n in [5,10,20,50,100,200,500,1000,5000,10000]:
        errors=[];sgd_errors=[]
        for repeat in range(10):
            X,y,w=synthetic(n=n,seed_value=42+repeat)
            theta=torch.linalg.lstsq(torch.cat([X,torch.ones(n,1)],1),y).solution.flatten()
            errors.append([float((theta[:2]-w).norm()),abs(theta[2].item()-4.2)])
            net,_=fit_regression(X,y,epochs=3,lr=.03)
            sgd_errors.append([float((net.weight.detach().flatten()-w).norm()),abs(net.bias.item()-4.2)])
        rows.append({"n":n,"sgd_w_error_mean":np.mean(sgd_errors,0)[0],"sgd_bias_error_mean":np.mean(sgd_errors,0)[1],
                     "ols_w_error_mean":np.mean(errors,0)[0],"ols_bias_error_mean":np.mean(errors,0)[1]})
    df=r.table("data_size",rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind,label in zip(axes,['w','bias'],['Weight error','Bias error']):
        ax.loglog(df.n,df[f'sgd_{kind}_error_mean'],'o-',label='SGD: 3 epochs, LR=0.03')
        ax.loglog(df.n,df[f'ols_{kind}_error_mean'],'o-',label='Exact least squares reference')
        ax.set(xlabel='Training samples',ylabel=label);ax.legend(fontsize=8)
    r.figure("parameter_error_vs_data")
    X,y,_=synthetic();net,trace=fit_regression(X,y,loss_kind="huber")
    r.note(f"Framework Huber final loss={trace[-1]:.7f}; weight.grad={net.weight.grad.tolist()}")
    r.note("Sample-size plot includes framework SGD with the book's LR=0.03 and 3 epochs, plus exact least squares as a separate reference. Ten data seeds per size. Larger n also means more SGD updates at fixed epochs.")


def experiment_3_6(r):
    X,y,w=synthetic(n=1000,noise=.5)
    Xv,yv,_=synthetic(n=1000,noise=.5,seed_value=777)
    rows=[]
    for n in [5,10,20,50,100,200,500,1000]:
        A=torch.cat([X[:n],torch.ones(n,1)],1)
        theta=torch.linalg.lstsq(A,y[:n]).solution
        pred=torch.cat([Xv,torch.ones(len(Xv),1)],1)@theta
        rows.append({"n":n,"train_mse":float(((A@theta-y[:n])**2).mean()),"validation_mse":float(((pred-yv)**2).mean())})
    df=r.table("learning_curve",rows)
    plt.figure(figsize=(7,4));plt.semilogx(df.n,df.train_mse,"o-",label="train");plt.semilogx(df.n,df.validation_mse,"o-",label="validation");plt.legend();r.figure("data_learning_curve")


def experiment_3_7(r):
    X,y,_=synthetic(n=120,d=200,noise=.01,bias=.05)
    Xt,yt,Xv,yv=X[:20],y[:20],X[20:],y[20:]
    # Theo sách: 20 train, 100 val, d=200, b=0.05, batch=5, LR=.01, 10 epoch.
    rows=[]
    for lam in [0,.01,.1,1,3,10]:
        seed();net=nn.Linear(200,1)
        with torch.no_grad():net.weight.normal_(0,.01);net.bias.zero_()
        optimizer=torch.optim.SGD([
            {'params':[net.weight],'weight_decay':lam},
            {'params':[net.bias],'weight_decay':0.},
        ],lr=.01)
        loader=DataLoader(TensorDataset(Xt,yt),batch_size=5,shuffle=True,generator=torch.Generator().manual_seed(42))
        for _ in range(10):
            for xb,yb in loader:
                optimizer.zero_grad();loss=((net(xb)-yb)**2).mean()/2
                loss.backward();optimizer.step()
        with torch.no_grad():
            rows.append({'lambda':lam,'train_mse':float(((net(Xt)-yt)**2).mean()),
                         'validation_mse':float(((net(Xv)-yv)**2).mean()),'weight_norm':float(net.weight.norm())})
    df=r.table('weight_decay',rows)
    plt.figure(figsize=(7,4));plt.plot(df['lambda'],df.train_mse,'o-',label='train MSE');plt.plot(df['lambda'],df.validation_mse,'o-',label='validation MSE');plt.xscale('symlog',linthresh=.01);plt.yscale('log');plt.legend();r.figure('regularization_curve')
    # Ridge chính xác chỉ làm tham chiếu; cùng mục tiêu half-MSE + lambda/2 ||w||².
    A=torch.cat([Xt,torch.ones(len(Xt),1)],1).double()
    Av=torch.cat([Xv,torch.ones(len(Xv),1)],1).double()
    penalty=torch.eye(201,dtype=torch.float64);penalty[-1,-1]=0
    rows=[]
    for lam in [0,1e-5,1e-4,.001,.01,.1,1,10,100]:
        if lam==0: theta=torch.linalg.lstsq(A,yt.double(),driver="gelsd").solution
        else: theta=torch.linalg.solve(A.T@A+len(A)*lam*penalty,A.T@yt.double())
        rows.append({"lambda":lam,"train_mse":float(((A@theta-yt)**2).mean()),"validation_mse":float(((Av@theta-yv)**2).mean()),"weight_norm":float(theta[:-1].norm())})
    r.table("exact_ridge_reference",rows)
    r.note("Regression has no classification accuracy: report unpenalized MSE. Main curve is SGD, LR=.01, batch=5, 10 epochs, true bias=.05, no bias penalty. Exact ridge is a separate reference, not the SGD result.")


def get_fashion():
    global FASHION
    if FASHION is None: FASHION=fashion()
    return FASHION


class ScratchSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.W=nn.Parameter(torch.randn(784,10)*.01)
        self.b=nn.Parameter(torch.zeros(10))
    def forward(self,X): return X@self.W+self.b


class ScratchMLP(nn.Module):
    def __init__(self,width=256):
        super().__init__()
        self.W1=nn.Parameter(torch.randn(784,width)*.01);self.b1=nn.Parameter(torch.zeros(width))
        self.W2=nn.Parameter(torch.randn(width,10)*.01);self.b2=nn.Parameter(torch.zeros(10))
    def forward(self,X):
        return torch.relu(X@self.W1+self.b1)@self.W2+self.b2


class GaussianNoise(nn.Module):
    def __init__(self,std=.15): super().__init__();self.std=std
    def forward(self,X): return X+torch.randn_like(X)*self.std if self.training else X


def make_classifier(hidden=(),activation="relu",dropout=(0.,0.),noise=False,initialization="default"):
    layers=[];previous=784
    for i,width in enumerate(hidden):
        layers.extend([nn.Linear(previous,width),{"relu":nn.ReLU,"tanh":nn.Tanh,"sigmoid":nn.Sigmoid}[activation]()])
        if noise: layers.append(GaussianNoise())
        if i<len(dropout) and dropout[i]: layers.append(nn.Dropout(dropout[i]))
        previous=width
    layers.append(nn.Linear(previous,10));net=nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer,nn.Linear):
            if initialization=="zero": nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
            elif initialization=="normal001": nn.init.normal_(layer.weight,std=.01);nn.init.zeros_(layer.bias)
            elif initialization=="xavier": nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
    return net


def classification_metrics(net,X,y):
    net.eval();loss=0.;correct=0
    with torch.no_grad():
        for i in range(0,len(X),2048):
            out=net(X[i:i+2048]);labels=y[i:i+2048]
            loss+=nn.functional.cross_entropy(out,labels,reduction="sum").item()
            correct+=(out.argmax(1)==labels).sum().item()
    return loss/len(X),correct/len(X)


def activation_variances(net,X):
    """Đo eval/train riêng; không đổi RNG hoặc chế độ của lần học tiếp theo."""
    values={}
    modes=[(module,module.training) for module in net.modules()]
    try:
        with torch.no_grad(),torch.random.fork_rng(devices=[]):
            net.eval();h=X;index=0
            for layer in net:
                h=layer(h)
                if isinstance(layer,(nn.ReLU,nn.Sigmoid,nn.Tanh)):
                    index+=1;values[f'h{index}_eval_var']=h.var(unbiased=False).item()
            net.train();h=X;index=0
            for layer in net:
                h=layer(h)
                if isinstance(layer,(nn.ReLU,nn.Sigmoid,nn.Tanh)):
                    index+=1
                    values[f'h{index}_before_dropout_var']=h.var(unbiased=False).item()
                    values[f'h{index}_after_dropout_var']=h.var(unbiased=False).item()
                elif isinstance(layer,nn.Dropout):
                    values[f'h{index}_after_dropout_var']=h.var(unbiased=False).item()
                elif isinstance(layer,GaussianNoise):
                    values[f'h{index}_after_noise_var']=h.var(unbiased=False).item()
    finally:
        for module,mode in modes:module.training=mode
    return values


def fit_classifier(name,hidden=(),lr=.1,batch=256,epochs=10,activation="relu",dropout=(0.,0.),decay=0.,noise=False,initialization="default",scratch=False):
    seed();X,y,Xv,yv=get_fashion()
    net=(ScratchMLP(hidden[0]) if hidden else ScratchSoftmax()) if scratch else make_classifier(hidden,activation,dropout,noise,initialization)
    opt=torch.optim.SGD(net.parameters(),lr=lr,weight_decay=decay)
    trace=[];g=torch.Generator().manual_seed(42);start=time.perf_counter()
    for epoch in range(1,epochs+1):
        net.train();permutation=torch.randperm(len(X),generator=g)
        for indices in permutation.split(batch):
            opt.zero_grad()
            logits=net(X[indices])
            if scratch and not hidden:
                shifted=logits-logits.max(dim=1,keepdim=True).values
                log_probs=shifted-shifted.exp().sum(dim=1,keepdim=True).log()
                loss=-log_probs[torch.arange(len(indices)),y[indices]].mean()
            else: loss=nn.functional.cross_entropy(logits,y[indices])
            loss.backward();opt.step()
        train_loss,train_acc=classification_metrics(net,X,y)
        val_loss,val_acc=classification_metrics(net,Xv,yv)
        # So sánh phương sai kích hoạt cùng 1.024 ảnh, theo từng tầng, trước/sau dropout.
        activation_vars={}
        if isinstance(net,nn.Sequential) and hidden:
            activation_vars=activation_variances(net,X[:1024])
        trace.append({"name":name,"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"train_accuracy":train_acc,"val_accuracy":val_acc,**activation_vars})
        if epoch in [1,epochs]: print(f"{name}: epoch {epoch}/{epochs}, val accuracy={val_acc:.4f}",flush=True)
    seconds=time.perf_counter()-start
    return {**trace[-1],"seconds":seconds,"hidden":str(hidden),"lr":lr,"batch":batch,"activation":activation,"dropout":str(dropout),"weight_decay":decay,"initialization":initialization,"scratch":scratch},trace,net


def classifier_sweep(r,configs):
    rows=[];traces=[]
    for config in configs:
        row,trace,_=fit_classifier(**config);rows.append(row);traces.extend(trace)
    table=r.table("comparison",rows);history=r.table("history",traces)
    plt.figure(figsize=(8,4))
    for name,group in history.groupby("name",sort=False): plt.plot(group.epoch,group.val_loss,label=name)
    plt.xlabel("Epoch");plt.ylabel("Validation cross-entropy");plt.legend(fontsize=7);r.figure("validation_learning_curves")
    return table,history


def experiment_4_1(r):
    logits=torch.tensor([.2,.5,1.1],dtype=torch.float64,requires_grad=True)
    p=logits.softmax(0);H=torch.autograd.functional.hessian(lambda x:torch.logsumexp(x,0)-x[1],logits)
    covariance=torch.diag(p)-p[:,None]*p[None,:]
    assert torch.allclose(H,covariance)
    r.table("hessian",H.detach().numpy());r.table("one_hot_covariance",covariance.detach().numpy())
    r.note(f"logsumexp(x+3)-logsumexp(x)={(torch.logsumexp(logits+3,0)-torch.logsumexp(logits,0)).item()}; softmax invariant={torch.allclose(logits.softmax(0),(logits+3).softmax(0))}")


def experiment_4_2(r):
    X,y,Xv,yv=get_fashion()
    r.note(f"Fashion-MNIST shapes: train={tuple(X.shape)}, labels={tuple(y.shape)}, test/validation={tuple(Xv.shape)}")
    fig,axes=plt.subplots(2,5,figsize=(9,4))
    for label,ax in enumerate(axes.flat): ax.imshow(X[(y==label).nonzero()[0]].reshape(28,28),cmap="gray");ax.set_title(str(label));ax.axis("off")
    r.figure("fashion_examples")
    rows=[]
    dataset=TensorDataset(X,y)
    for batch in [1,32,256,1024]:
        t=time.perf_counter();count=0
        for xb,yb in DataLoader(dataset,batch_size=batch,num_workers=0): count+=len(xb)
        seconds=time.perf_counter()-t;rows.append({"batch":batch,"samples":count,"seconds":seconds,"samples_per_second":count/seconds})
    r.table("data_iterator_speed",rows)
    t=time.perf_counter();count=0
    for xb,yb in zip(X.split(256),y.split(256)):count+=len(xb)
    elapsed=time.perf_counter()-t
    r.table("batch_fetch_optimization",[{"method":"tensor split views, already in RAM","samples":count,
        "seconds":elapsed,"samples_per_second":count/elapsed}])
    profiler=cProfile.Profile();profiler.enable()
    for _ in DataLoader(dataset,batch_size=256,num_workers=0): pass
    profiler.disable();buffer=io.StringIO();pstats.Stats(profiler,stream=buffer).sort_stats("cumtime").print_stats(12)
    r.note(buffer.getvalue())
    r.note("API datasets include MNIST, FashionMNIST, CIFAR10/100, ImageFolder, SVHN (torchvision docs). Here IDX is read directly to avoid torch/torchvision ABI mismatch; no data substitution.")


def experiment_4_3(r):
    values=torch.arange(11.).double()
    groups=values.split(4);quick=torch.stack([g.mean() for g in groups]).mean()
    exact=values.mean();last=groups[-1].mean();B=len(groups);m=4;n=len(values);b=len(groups[-1])
    corrected=(m*B*quick-(m-b)*last)/n
    assert torch.allclose(corrected,exact)
    r.table("batch_average",[{"n":n,"batch_size":m,"last_batch_size":b,"quick_mean":quick.item(),"weighted_mean":exact.item(),"corrected_mean":corrected.item()}])


def experiment_4_4(r):
    rows=[]
    for vals in [[100.,101.,102.],[-101.,-102.,-103.]]:
        x=torch.tensor(vals);naive=x.exp()/x.exp().sum();stable=(x-x.max()).exp();stable/=stable.sum()
        rows.append({"logits":str(vals),"naive":str(naive.tolist()),"stable":str(stable.tolist())})
    r.table("softmax_stability",rows)
    logits=torch.randn(1024,10);labels=torch.randint(10,(1024,));one_hot=nn.functional.one_hot(labels,10)
    dense=-(one_hot*logits.log_softmax(1)).sum(1).mean();fast=nn.functional.cross_entropy(logits,labels)
    assert torch.allclose(dense,fast)
    r.note(f"Dense one-hot CE={dense.item():.6f}; fused/index CE={fast.item():.6f}; negative sign is required.")
    rows=[]
    for label,fn in [("dense_one_hot",lambda:-(one_hot*logits.log_softmax(1)).sum(1).mean()),("index_fused",lambda:nn.functional.cross_entropy(logits,labels))]:
        fn();t=time.perf_counter()
        for _ in range(300):fn()
        rows.append({"implementation":label,"seconds_per_call":(time.perf_counter()-t)/300})
    r.table("cross_entropy_speed",rows)
    configs=[{"name":f"lr_{lr}","lr":lr} for lr in [.01,.1,1.]]
    configs += [{"name":f"batch_{batch}","batch":batch} for batch in [32,1024]]
    configs=[{**config,"scratch":True} for config in configs]
    classifier_sweep(r,configs)


def experiment_4_5(r):
    rows=[]
    for dtype in [torch.float64,torch.float32,torch.bfloat16,torch.float16]:
        info=torch.finfo(dtype)
        rows.append({"dtype":str(dtype),"log_max_finite":math.log(info.max),"log_min_normal":math.log(info.tiny),
                     "log_min_subnormal_theoretical":math.log(info.tiny)+math.log(info.eps)})
    r.table("exp_dynamic_range",rows)
    measured=[]
    for dtype in [torch.float64,torch.float32,torch.bfloat16,torch.float16]:
        low,high=-2000.,0.
        for _ in range(80):
            middle=(low+high)/2
            if torch.exp(torch.tensor(middle,dtype=dtype)).item()>0:high=middle
            else:low=middle
        minimum=torch.tensor(high,dtype=dtype)
        low,high=0.,2000.
        for _ in range(80):
            middle=(low+high)/2
            if torch.isfinite(torch.exp(torch.tensor(middle,dtype=dtype))):low=middle
            else:high=middle
        maximum=torch.tensor(low,dtype=dtype)
        below=torch.nextafter(minimum,torch.tensor(float('-inf'),dtype=dtype))
        above=torch.nextafter(maximum,torch.tensor(float('inf'),dtype=dtype))
        assert torch.exp(minimum).item()>0 and torch.exp(below).item()==0
        assert torch.isfinite(torch.exp(maximum)) and not torch.isfinite(torch.exp(above))
        measured.append({'dtype':str(dtype),'smallest_representable_input_with_nonzero_exp':minimum.item(),
                         'largest_representable_input_with_finite_exp':maximum.item(),
                         'exp_at_previous_input':torch.exp(below).item(),'exp_at_next_input':'overflow'})
    r.table('exp_measured_boundaries_cpu',measured)
    classifier_sweep(r,[{"name":"softmax_longer_20epochs","epochs":20}])
    r.note("LR curves reuse section 4.4, avoiding duplicate runs. FP16/BF16 subnormals may be flushed to zero by hardware. TF32 uses FP32 exponent but is a CUDA matmul mode, not CPU exp dtype.")


def experiment_4_6(r):
    n=math.ceil(math.log(2/.001)/(2*.0001**2))
    r.note(f"Hoeffding sufficient n={n:,} for epsilon=0.0001, delta=0.001, fixed classifier and IID test draws.")
    labels=np.array([0,1,1,0,1,0]);baseline=np.zeros(len(labels),dtype=int)
    base_err=np.mean(baseline!=labels);recovered=[]
    for i in range(len(labels)):
        p=baseline.copy();p[i]=1
        recovered.append(int(np.mean(p!=labels)<base_err))
    assert np.array_equal(recovered,labels)
    r.note(f"Exact-error test oracle: baseline+N one-coordinate queries recovered labels {recovered}.")


def experiment_4_7(r):
    rng=np.random.default_rng(42);source=rng.normal(0,1,(4000,2));target=rng.normal(.8,1,(4000,2))
    X=np.vstack([source,target]);domain=np.r_[np.zeros(len(source)),np.ones(len(target))]
    a,b=train_test_split(np.arange(len(X)),test_size=.3,random_state=42,stratify=domain)
    detector=LogisticRegression().fit(X[a],domain[a]);auc=roc_auc_score(domain[b],detector.predict_proba(X[b])[:,1])
    # Dùng riêng source hold-out; không đánh giá trên dữ liệu đã fit detector.
    held=b[b<len(source)];prob=detector.predict_proba(X[held])[:,1]
    weights=np.clip(prob/(1-prob),.01,20)
    y=source[:,0]+source[:,1]
    expected=target[:,0].mean()+target[:,1].mean()
    r.table("shift_detection_correction",[{"heldout_domain_auc":auc,"source_mean":y[held].mean(),
        "importance_weighted_mean":np.average(y[held],weights=weights),"target_mean":expected,
        "effective_sample_size":weights.sum()**2/(weights@weights)}])
    r.note("Synthetic covariate shift with shared conditional y=x1+x2. Domain priors are equal; weights=p(target|x)/p(source|x). Correction needs overlap and unchanged p(y|x), not label/concept shift.")


def experiment_5_1(r):
    x=torch.tensor([-30.,-10.,0.,10.,30.],requires_grad=True);x.sigmoid().sum().backward()
    r.table("sigmoid_gradients",[{"x":a,"gradient":b} for a,b in zip(x.detach().tolist(),x.grad.tolist())])
    values=torch.linspace(-3,3,100)
    assert torch.allclose(values.tanh()+1,2*(2*values).sigmoid(),atol=1e-6)
    r.note("tanh(x)+1=2 sigmoid(2x) verified numerically.")


def experiment_5_2(r):
    configs=[{"name":f"width_{w}","hidden":(w,)} for w in [1,64,128,256,512]]
    configs += [
        {"name":"two_layers","hidden":(256,256)},
        {"name":"lr_0.01","hidden":(256,),"lr":.01},
        {"name":"lr_0.5","hidden":(256,),"lr":.5},
        {"name":"tanh","hidden":(256,),"activation":"tanh"},
        {"name":"sigmoid","hidden":(256,),"activation":"sigmoid"},
        {"name":"zero_init","hidden":(256,),"initialization":"zero"},
        {"name":"normal001_init","hidden":(256,),"initialization":"normal001"},
        {"name":"xavier_init","hidden":(256,),"initialization":"xavier"},
        {"name":"scratch_width256","hidden":(256,),"scratch":True},
        {"name":"joint_layers128_lr0.1_15epochs","hidden":(128,128),"epochs":15},
        {"name":"joint_layers256_lr0.05_15epochs","hidden":(256,256),"lr":.05,"epochs":15},
    ]
    table,history=classifier_sweep(r,configs)
    width_table=table[table.name.str.startswith("width_")]
    plt.figure(figsize=(7,4));plt.plot([1,64,128,256,512],width_table.val_accuracy,"o-");plt.xlabel("Hidden neurons");plt.ylabel("Validation accuracy");r.figure("neurons_vs_accuracy")
    rows=[]
    for n in [1024,1025,1026,1028,1032]:
        a,b=torch.randn(n,n),torch.randn(n,n)
        rows.append({"size":n,"cpu_seconds":benchmark_matmul(a,b,repeats=5)})
    r.table("matrix_alignment_cpu",rows)
    r.note(f"CUDA available={torch.cuda.is_available()}. No GPU timing or physical memory-bus width is invented. CPU word size {platform.architecture()[0]} is not memory bus width; see hardware.json.")
    r.note("From-scratch vs framework speeds above include initialization differences. A separate equal-weight forward/backward benchmark follows.")
    speed=[]
    for width in [64,256,512]:
        seed();framework=make_classifier((width,),initialization="normal001");scratch=ScratchMLP(width)
        with torch.no_grad():
            scratch.W1.copy_(framework[0].weight.T);scratch.b1.copy_(framework[0].bias)
            scratch.W2.copy_(framework[2].weight.T);scratch.b2.copy_(framework[2].bias)
        X,y,_,_=get_fashion();xb,yb=X[:256],y[:256]
        assert torch.allclose(framework(xb),scratch(xb),atol=1e-5)
        for label,net in [("framework",framework),("scratch",scratch)]:
            for _ in range(3):net.zero_grad();nn.functional.cross_entropy(net(xb),yb).backward()
            t=time.perf_counter()
            for _ in range(20):net.zero_grad();nn.functional.cross_entropy(net(xb),yb).backward()
            speed.append({"width":width,"implementation":label,"forward_backward_seconds":(time.perf_counter()-t)/20})
    r.table("equal_weights_speed",speed)


def experiment_5_3(r):
    X=torch.randn(5,3,requires_grad=True);W1=torch.randn(3,4,requires_grad=True);b1=torch.randn(4,requires_grad=True);W2=torch.randn(4,2,requires_grad=True)
    Z=X@W1+b1;H=Z.relu();O=H@W2;loss=(O**2).mean()
    loss.backward();r.note(f"X shape={tuple(X.shape)}, gradient shape={tuple(X.grad.shape)}, b1 gradient shape={tuple(b1.grad.shape)}")
    r.note("Graph: X,W1,b1 -> Z -> ReLU -> H; H,W2 -> O; O,Y -> loss; weight matrices -> L2 -> objective.")
    fig,ax=plt.subplots(figsize=(11,4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    nodes={'X':(.05,.6),'W1':(.17,.9),'b1':(.33,.9),'Z=XW1+b1':(.25,.6),
           'H=ReLU(Z)':(.46,.6),'W2':(.62,.9),'O=HW2':(.68,.6),
           'Y':(.9,.9),'L(O,Y)':(.9,.6),'R(W1,W2)':(.46,.15),'J=L+R':(.9,.15)}
    for label,(a,b) in nodes.items():ax.text(a,b,label,ha='center',va='center',bbox={'boxstyle':'round','facecolor':'#e7eef7'})
    for start,end in [('X','Z=XW1+b1'),('W1','Z=XW1+b1'),('b1','Z=XW1+b1'),('Z=XW1+b1','H=ReLU(Z)'),
                      ('H=ReLU(Z)','O=HW2'),('W2','O=HW2'),('O=HW2','L(O,Y)'),('Y','L(O,Y)'),
                      ('L(O,Y)','J=L+R'),('R(W1,W2)','J=L+R')]:
        ax.annotate('',xy=nodes[end],xytext=nodes[start],arrowprops={'arrowstyle':'->','shrinkA':35,'shrinkB':40})
    for start in ['W1','W2']:
        ax.annotate('',xy=nodes['R(W1,W2)'],xytext=nodes[start],
                    arrowprops={'arrowstyle':'->','shrinkA':15,'shrinkB':35,'linestyle':'--','color':'gray','connectionstyle':'arc3,rad=0.2'})
    r.figure('computational_graph')
    d,h,q,N=784,256,10,32
    r.table('memory_accounting',[{'model':label,'parameters':P,'parameter_bytes_fp32':4*P,
        'parameter_plus_gradient_bytes_fp32':8*P,'stored_X_Z_H_O_bytes_fp32':4*N*(d+2*h+q)}
        for label,P in [('original_no_bias',d*h+h*q),('hidden_bias_added',d*h+h+h*q)]])
    x=torch.tensor([1.,2.],requires_grad=True);f=(x**3).sum();g=torch.autograd.grad(f,x,create_graph=True)[0]
    hv=torch.autograd.grad(g.sum(),x)[0];r.note(f"Second-derivative Hessian-vector product={hv.tolist()}")


def experiment_5_4(r):
    A=torch.tensor([[1.,100.],[0.,1.]],dtype=torch.float64);B=A.T
    r.note(f"Eigenvalues A={torch.linalg.eigvals(A).real.tolist()}, B={torch.linalg.eigvals(B).real.tolist()}, AB={torch.linalg.eigvals(A@B).real.tolist()}")
    r.note("Individual eigenvalues do not control a non-normal product; singular-value bounds are the safe general gradient bounds.")


def experiment_5_5(r):
    df=pd.read_csv(RESULTS/'4_5'/'history.csv')
    best=df.loc[df.val_loss.idxmin()];last=df.iloc[-1]
    r.table("early_stopping",[{"selected_epoch":int(best.epoch),"selected_val_loss":best.val_loss,"selected_val_accuracy":best.val_accuracy,
                                "final_epoch":int(last.epoch),"final_val_loss":last.val_loss,"final_val_accuracy":last.val_accuracy}])
    r.note("Checkpoint is selected from a measured curve, not assumed to outperform final training. The curve is softmax from section 4.5, used only to demonstrate the criterion.")


def experiment_5_6(r):
    configs=[
        {"name":"no_dropout","hidden":(256,256),"epochs":20},
        {"name":"dropout_02_05","hidden":(256,256),"dropout":(.2,.5),"epochs":20},
        {"name":"dropout_05_02","hidden":(256,256),"dropout":(.5,.2),"epochs":20},
        {"name":"book_dropout_05_05","hidden":(256,256),"dropout":(.5,.5)},
        {"name":"weight_decay","hidden":(256,256),"decay":1e-4,"epochs":20},
        {"name":"dropout_plus_decay","hidden":(256,256),"dropout":(.2,.5),"decay":1e-4,"epochs":20},
        {"name":"gaussian_noise","hidden":(256,256),"noise":True,"epochs":20},
    ]
    table,history=classifier_sweep(r,configs)
    rows=[]
    for name in ["no_dropout","dropout_02_05","dropout_05_02"]:
        subset=history[history.name==name]
        for h in [1,2]:
            col=f"h{h}_eval_var"
            plt.figure(figsize=(7,4));plt.plot(subset.epoch,subset[col],label="eval activation variance")
            after=f"h{h}_after_dropout_var"
            if after in subset and subset[after].notna().any():plt.plot(subset.epoch,subset[after],label="dropout training realization")
            plt.xlabel("Epoch");plt.ylabel("Variance across samples and units");plt.legend();r.figure(f"activation_variance_{name}_layer{h}")
        rows.append({"name":name,"epoch10_accuracy":subset[subset.epoch==10].val_accuracy.iloc[0],"epoch20_accuracy":subset[subset.epoch==20].val_accuracy.iloc[0]})
    r.table("longer_training",rows)
    r.note("Noise/dropout can help or hurt; compare measured validation scores, no promise Gaussian noise wins. Eval and stochastic train variance are measured in separate forwards on 1024 fixed images. RNG and module modes are restored, so monitoring does not alter later training masks. Empirical variance across samples and units differs from conditional mask variance p/(1-p)*h^2.")


def experiment_5_7(r):
    from house_prices import run_house_prices
    run_house_prices(r)


EXPERIMENTS = {name.replace("experiment_","").replace("_","."): fn
               for name,fn in list(globals().items()) if name.startswith("experiment_")}


def run_section(section,recompute=False):
    path=RESULTS/section.replace(".","_")/"result.json"
    if path.is_file() and not recompute:
        return json.loads(path.read_text(encoding="utf-8"))
    seed();record=Record(section)
    EXPERIMENTS[section](record)
    path.write_text(json.dumps(record.result,ensure_ascii=False,indent=2),encoding="utf-8")
    return record.result


def render_result(result):
    from IPython.display import display,Image,Markdown
    for note in result['notes']: print(note)
    for name in result['tables']:
        display(Markdown(f"**{Path(name).stem}**"))
        display(pd.read_csv(BASE/name).head(40))
    for name in result['figures']:display(Image(filename=str(BASE/name)))


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--sections",nargs="+",default=list(EXPERIMENTS))
    parser.add_argument("--recompute",action="store_true")
    args=parser.parse_args()
    for section in args.sections:
        print("Running",section,flush=True);run_section(section,args.recompute)
