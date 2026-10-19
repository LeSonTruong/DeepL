"""Kiểm tra nội dung bằng PDF gốc, các đồng nhất thức và kết quả CSV độc lập.

Không dùng việc notebook chạy không lỗi làm bằng chứng rằng mọi câu trả lời đúng.
Chạy từ Python có requirements: python verify_results.py
"""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader
import torch
from torch import nn
from build_submission import COUNTS,PAGES
from data import BASE
from experiments import activation_variances,make_classifier,seed,synthetic


def check_pdf():
    pdf=BASE.parent/'d2l-en.pdf'
    reader=PdfReader(pdf);flat=[]
    inventory={item['pdf_page']:item['text'] for item in json.loads(
        (BASE/'source_reference/exercises_inventory.json').read_text(encoding='utf-8'))}
    def walk(items):
        for item in items:
            if isinstance(item,list):walk(item)
            else:flat.append((item.title,reader.get_destination_page_number(item)))
    walk(reader.outline)
    rows=[]
    for i,(title,page) in enumerate(flat):
        if title!='Exercises' or not 40<=page<246:continue
        section=next(key for key,value in PAGES.items() if value==page+1)
        end=flat[i+1][1]
        text='\n'.join(reader.pages[p].extract_text() for p in range(page,min(end+1,page+4)))
        text=text[text.find('Exercises'):].split('Discussions')[0]
        assert text.strip()==inventory[page+1].strip(),f'Cached question differs from original PDF: {section}'
        # PDF có dòng "1032." là kích thước ma trận 5.2, không phải số thứ tự câu.
        indices=[int(value) for value in re.findall(r'(?m)^(\d+)\.\s',text) if int(value)<100]
        assert max(indices)==COUNTS[section],(section,indices)
        assert set(range(1,COUNTS[section]+1))<=set(indices)
        rows.append({'section':section,'pdf_page':page+1,'max_question_index':max(indices)})
    assert len(rows)==28 and sum(item['max_question_index'] for item in rows)==162
    pd.DataFrame(rows).to_csv(BASE/'audit_pdf_questions.csv',index=False)
    return {'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'exercise_groups':len(rows),'main_questions':162}


def check_math():
    x=torch.tensor([.2,.5,1.1],dtype=torch.float64,requires_grad=True)
    H=torch.autograd.functional.hessian(lambda z:torch.logsumexp(z,0)-z[1],x)
    p=x.softmax(0);cov=torch.diag(p)-p[:,None]*p[None,:]
    assert torch.allclose(H,cov) and torch.linalg.eigvalsh(H).min()>-1e-12
    assert torch.allclose(torch.logsumexp(x+3,0),torch.logsumexp(x,0)+3)
    assert torch.allclose(x.softmax(0),(x+3).softmax(0))
    h=1e-6;f=lambda z:z**3-1/z
    assert math.isclose((f(1+h)-f(1-h))/(2*h),4.,abs_tol=1e-8)
    codes=['000','001','010','011','100','101','110','1110','1111']
    assert all(not b.startswith(a) for a in codes for b in codes if a!=b)
    assert math.isclose(sum(2**(-len(c)) for c in codes),1.)
    assert math.isclose(sum(map(len,codes))/18,29/18)
    assert math.ceil(math.log(2000)/(2e-8))==380045123
    # Kiểm tra phương trình backward có hidden bias và L2 không phạt bias.
    torch.manual_seed(7)
    X=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
    W1=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    b1=torch.randn(4,dtype=torch.float64,requires_grad=True)
    W2=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
    labels=torch.tensor([0,1,0,1,1]);lam=.03
    Z=X@W1+b1;H=Z.relu();O=H@W2
    loss=nn.functional.cross_entropy(O,labels)+lam/2*(W1.square().sum()+W2.square().sum())
    grads=torch.autograd.grad(loss,(X,W1,b1,W2))
    G=(O.softmax(1)-nn.functional.one_hot(labels,2))/len(X)
    D=(G@W2.T)*(Z>0)
    expected=(D@W1.T,X.T@D+lam*W1,D.sum(0),H.T@G+lam*W2)
    assert all(torch.allclose(a,b,atol=1e-10) for a,b in zip(grads,expected))
    return 'CE Hessian/covariance, tangent, Huffman, Hoeffding and MLP backward verified'


def check_monitoring():
    seed();net=make_classifier((4,3),dropout=(.2,.5));X=torch.randn(32,784)
    net.train();modes=[m.training for m in net.modules()];rng=torch.get_rng_state().clone()
    measured=activation_variances(net,X)
    assert torch.equal(torch.get_rng_state(),rng),'Monitoring consumed training RNG'
    assert modes==[m.training for m in net.modules()],'Monitoring changed module modes'
    captures=[]
    handles=[m.register_forward_hook(lambda _m,_x,out:captures.append(out.detach().clone()))
             for m in net if isinstance(m,nn.ReLU)]
    net.eval()
    with torch.no_grad():net(X)
    for handle in handles:handle.remove()
    for index,h in enumerate(captures,1):
        assert math.isclose(measured[f'h{index}_eval_var'],h.var(unbiased=False).item(),rel_tol=1e-6)
    return 'Independent eval hooks agree at both layers; RNG and train/eval modes preserved'


def check_csv():
    for folder in (BASE/'results').iterdir():
        if not folder.is_dir():continue
        if (folder/'comparison.csv').is_file() and (folder/'history.csv').is_file():
            rows=pd.read_csv(folder/'comparison.csv');history=pd.read_csv(folder/'history.csv')
            for row in rows.to_dict('records'):
                trace=history[history.name==row['name']]
                assert trace.epoch.tolist()==list(range(1,int(row['epoch'])+1))
                for metric in ['train_loss','val_loss','train_accuracy','val_accuracy']:
                    assert math.isfinite(row[metric])
                    assert np.isclose(row[metric],trace.iloc[-1][metric])
                assert 0<=row['val_accuracy']<=1
    config=pd.read_csv(BASE/'results/3_4/initialization.csv')
    assert np.allclose(config.init_variance,config.init_std**2)
    assert np.isclose(config.init_variance,.01).any()
    X,y,w=synthetic(n=120,d=200,bias=.05)
    assert abs((y.flatten()-X@w).mean().item()-.05)<.005
    weights=pd.read_csv(BASE/'results/3_7/weight_decay.csv')
    assert weights['lambda'].tolist()==[0,.01,.1,1,3,10]
    assert np.isfinite(weights.select_dtypes('number')).all().all()
    folds=pd.read_csv(BASE/'results/5_7/kfold_scores.csv')
    summary=pd.read_csv(BASE/'results/5_7/model_selection.csv')
    for row in summary.to_dict('records'):
        scores=folds[folds.name==row['name']].val_log_rmse
        assert len(scores)==5 and np.isfinite(scores).all()
        assert np.isclose(scores.mean(),row['cv_log_rmse_mean'])
        assert np.isclose(scores.std(),row['cv_log_rmse_std'])
    sub=pd.read_csv(BASE/'Le_Son_Truong_submission.csv')
    test=pd.read_csv(BASE/'data/kaggle_house_pred_test.csv')
    assert sub.columns.tolist()==['Id','SalePrice'] and len(sub)==1459
    assert np.array_equal(sub.Id,test.Id) and np.isfinite(sub.SalePrice).all() and (sub.SalePrice>0).all()
    for item in json.loads((BASE/'data_manifest.json').read_text()):
        assert hashlib.sha256((BASE/'data'/item['file']).read_bytes()).hexdigest()==item['sha256']
    return 'CSV/history consistency, initialization variance, weight-decay data, CV aggregates, submission IDs and dataset SHA-256 verified'


if __name__=='__main__':
    report={'source':check_pdf(),'math':check_math(),'dropout_monitoring':check_monitoring(),'results':check_csv(),
            'scope':'Automated identities and data checks plus manual question review; not a proof that every empirical conclusion generalizes.',
            'pending':['5.2.7 GPU timing with CUDA','5.7.1 real Kaggle submission/score','Drive upload by user']}
    (BASE/'audit_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Audit checks passed. Details: audit_checks.json')
