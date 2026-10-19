"""D2L 5.7: preprocessing trong fold, K-fold, tuning và CSV nộp Kaggle."""
import json
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
from data import DATA, BASE


def preprocess(train,other,standardize=True):
    numeric=train.select_dtypes(include="number").columns
    categorical=train.columns.difference(numeric)
    means=train[numeric].mean()
    std=train[numeric].std().replace(0,1).fillna(1)
    def transform(frame):
        numbers=frame[numeric].fillna(means)
        if standardize: numbers=(numbers-means)/std
        cats=pd.get_dummies(frame[categorical].fillna("<MISSING>").astype(str),dtype=float)
        return pd.concat([numbers,cats],axis=1).astype(np.float32)
    a=transform(train)
    b=transform(other).reindex(columns=a.columns,fill_value=0.)
    assert np.isfinite(a.values).all() and np.isfinite(b.values).all()
    return torch.tensor(a.values,dtype=torch.float32),torch.tensor(b.values,dtype=torch.float32)


def fit(X,y,Xv,yv,config):
    torch.manual_seed(42)
    layers=[];previous=X.shape[1]
    for width in config['hidden']:
        layers.extend([nn.Linear(previous,width),nn.ReLU()])
        if config['dropout']:layers.append(nn.Dropout(config['dropout']))
        previous=width
    layers.append(nn.Linear(previous,1));net=nn.Sequential(*layers)
    optimizer_type=torch.optim.SGD if config['optimizer']=='sgd' else torch.optim.Adam
    optimizer=optimizer_type(net.parameters(),lr=config['lr'],weight_decay=config['decay'])
    generator=torch.Generator().manual_seed(42)
    for _ in range(config['epochs']):
        net.train()
        for idx in torch.randperm(len(X),generator=generator).split(64):
            optimizer.zero_grad();loss=((net(X[idx])-y[idx])**2).mean()/2
            if not torch.isfinite(loss):return None,float('nan'),float('nan')
            loss.backward();optimizer.step()
    net.eval()
    with torch.no_grad():
        train_rmse=((net(X)-y)**2).mean().sqrt().item()
        val_rmse=((net(Xv)-yv)**2).mean().sqrt().item()
    if not math.isfinite(val_rmse):return None,train_rmse,val_rmse
    return net,train_rmse,val_rmse


def run_house_prices(r):
    train=pd.read_csv(DATA/'kaggle_house_pred_train.csv')
    test=pd.read_csv(DATA/'kaggle_house_pred_test.csv')
    features=train.drop(columns=['Id','SalePrice']);test_features=test.drop(columns='Id')
    labels=torch.tensor(np.log(train.SalePrice.values),dtype=torch.float32).reshape(-1,1)
    base={'hidden':(), 'lr':.01,'epochs':10,'optimizer':'sgd','decay':0.,'dropout':0.}
    configs=[
        {'name':'book_linear_10epochs',**base},
        {'name':'linear_100epochs',**base,'epochs':100},
        {'name':'ridge_100epochs',**base,'epochs':100,'decay':.01},
        {'name':'mlp64_dropout',**base,'hidden':(64,),'epochs':100,'optimizer':'adam','lr':.003,'dropout':.1,'decay':1e-4},
        {'name':'mlp64_32_dropout',**base,'hidden':(64,32),'epochs':100,'optimizer':'adam','lr':.003,'dropout':.1,'decay':1e-3},
    ]
    folds=list(KFold(5,shuffle=True,random_state=42).split(features))
    fold_rows=[];summary=[];best_score=float('inf');best=[];best_config=None
    for config in configs:
        scores=[];models=[]
        for fold,(tr,va) in enumerate(folds):
            X,Xv=preprocess(features.iloc[tr],features.iloc[va])
            net,train_error,val_error=fit(X,labels[tr],Xv,labels[va],config)
            fold_rows.append({'name':config['name'],'fold':fold+1,'train_log_rmse':train_error,'val_log_rmse':val_error})
            scores.append(val_error);models.append((net,tr))
        score=float(np.mean(scores))
        summary.append({**config,'cv_log_rmse_mean':score,'cv_log_rmse_std':float(np.std(scores,ddof=1))})
        print(f"House prices {config['name']}: 5-fold log RMSE={score:.5f}",flush=True)
        if math.isfinite(score) and score<best_score:best_score=score;best=models;best_config=config
    r.table('kfold_scores',fold_rows);table=r.table('model_selection',summary)
    r.table('data_summary',[{'train_rows':len(train),'unlabeled_test_rows':len(test),'raw_features':features.shape[1],
                             'missing_cell_fraction':features.isna().to_numpy().mean()}])
    # Thí nghiệm không scale cùng baseline, ghi rõ có thể phân kỳ thay vì bịa một điểm tốt.
    tr,va=folds[0];X,Xv=preprocess(features.iloc[tr],features.iloc[va],standardize=False)
    net,train_error,val_error=fit(X,labels[tr],Xv,labels[va],configs[0])
    r.table('without_standardization',[{'standardize':False,'fold':1,'train_log_rmse':train_error,'val_log_rmse':val_error,
                                      'status':'diverged' if net is None else 'finite'}])
    predictions=[]
    for net,tr in best:
        _,Xt=preprocess(features.iloc[tr],test_features)
        net.eval()
        with torch.no_grad():predictions.append(net(Xt).flatten().exp().numpy())
    predicted=np.mean(predictions,axis=0)
    assert len(predicted)==1459 and np.isfinite(predicted).all() and (predicted>0).all()
    submission=pd.DataFrame({'Id':test.Id,'SalePrice':predicted})
    output=BASE/'Le_Son_Truong_submission.csv';submission.to_csv(output,index=False)
    r.table('submission_preview',submission.head(10))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,4));plt.bar(table.name,table.cv_log_rmse_mean);plt.xticks(rotation=25,ha='right');plt.ylabel('5-fold log RMSE');r.figure('kfold_model_comparison')
    r.note(f"Selected config={best_config['name']}; 5-fold log RMSE={best_score:.5f}; submission={output.name}, {len(submission)} rows.")
    r.note("Kaggle submission has NOT been uploaded. Leaderboard score is unknown; local CV score is not a Kaggle score. User must accept competition rules and upload via own account.")
    (r.path/'selected_config.json').write_text(json.dumps(best_config,indent=2),encoding='utf-8')
