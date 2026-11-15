import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
import config as cfg
from scripts.run_v3_dev import MECHS, SEEDS, data_dir
from scripts.generate_paper_simulation import FEATURE_LAW
from src.data_pipeline import prepare_labels, stratified_split
from src.metrics import compute_all_metrics
from sklearn.linear_model import LogisticRegression
def con(df,f):
    cats,_,vals=FEATURE_LAW[f]; return df[f].map(dict(zip(cats,vals))).astype(float).values
def X(df,pairs):
    cols=["rider_type","age_group","gender","traffic_condition","police_presence","weather","time_slot","weekend"]
    A=[con(df,c) for c in cols]+[df.num_lanes.values-3.0, np.where(df.intersection_id%2==0,1.0,-1.0)]
    if pairs: A+= [con(df,"rider_type")*con(df,"traffic_condition"),con(df,"age_group")*con(df,"police_presence"),con(df,"rider_type")*con(df,"weather")]
    return np.column_stack(A+[np.eye(12)[df.intersection_id.values]])
for m in MECHS:
  r={}
  for pairs in (False,True):
    a=[]
    for s in SEEDS:
      raw=pd.read_csv(data_dir(m,s)/"data.csv"); raw["_row_id"]=np.arange(len(raw))
      tr,va,te=stratified_split(prepare_labels(raw,cfg.LABEL_POLICY,True),seed=42)
      P=np.zeros((len(va),4));Y=np.zeros((len(va),4));M=np.zeros((len(va),4))
      for k,t in enumerate(cfg.TASK_NAMES):
        mk=cfg.TASK_MASK_COLS[t]; sel=tr[mk].astype(bool).values
        clf=LogisticRegression(C=1.0,max_iter=3000).fit(X(tr,pairs)[sel],tr.loc[sel,t])
        P[:,k]=clf.predict_proba(X(va,pairs))[:,1];Y[:,k]=va[t].fillna(0).values;M[:,k]=va[mk].values
      a.append(compute_all_metrics(Y,P,M)["macro"]["roc_auc"])
    r[pairs]=round(np.mean(a),4)
  print(m,r)
