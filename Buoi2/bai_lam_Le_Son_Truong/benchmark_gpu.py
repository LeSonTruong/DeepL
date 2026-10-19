"""Bổ sung phép đo GPU cho 5.2 câu 7 khi kernel có PyTorch CUDA."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch


def benchmark():
    rows=[]
    devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        for n in [1024,1025,1026,1028,1032]:
            A,B=torch.randn(n,n,device=device),torch.randn(n,n,device=device)
            for _ in range(3):A@B
            times=[]
            for _ in range(10):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();A@B
                if device=='cuda':torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            seconds=float(np.median(times))
            rows.append({'device':device,'size':n,'median_seconds':seconds,'GFLOP_s':2*n**3/seconds/1e9})
    frame=pd.DataFrame(rows)
    target=Path(__file__).resolve().parent/'results'/'5_2'/'cpu_gpu_benchmark.csv'
    target.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(target,index=False)
    print(frame.to_string(index=False))
    if 'cuda' not in devices:print('GPU timing not measured: this Python environment has no CUDA support.')
    return frame


if __name__=='__main__':benchmark()
