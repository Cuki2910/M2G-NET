"""Post-hoc supplement: macro AUC of each procedure as a 2-member (training seeds 42+101) probability average,
scored on the eligible population as in results-locked.csv. No refits. Run from repo root."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import sys; sys.path.insert(0, ".")
import config as cfg
u='outputs/paper-v3-locked/units/'; dd='outputs/paper-v3-locked/data/'
P=pd.read_csv('docs/paper/extension-v3/per-set-locked.csv')
mods=['single_task_mlp','early_fusion_mlp','logistic_regression','catboost','m2g_net_v3']
def macro(pred,oracle):
    v=[]
    for t in cfg.TASK_NAMES:
        s=pred[pred.task==t].merge(oracle,left_on='row_id',right_on='simulation_row_id',validate='one_to_one')
        e=s[t+'_eligible']==1
        v.append(roc_auc_score(s.loc[e,t+'_latent'],s.loc[e,'p']))
    return float(np.mean(v))
rows=[]
for mech in ['additive','interaction','partial']:
    for s in range(1711,1731):
        oracle=pd.read_csv(f'{dd}{mech}-{s}/oracle.csv')
        for m in mods:
            a=pd.read_csv(f'{u}{mech}-g{s}-t42-{m}/predictions.csv').sort_values(['task','row_id']).reset_index(drop=True)
            b=pd.read_csv(f'{u}{mech}-g{s}-t101-{m}/predictions.csv').sort_values(['task','row_id']).reset_index(drop=True)
            assert (a.row_id==b.row_id).all() and (a.task==b.task).all()
            pa=a.assign(p=a.raw_probability);pb=b.assign(p=b.raw_probability);pe=a.assign(p=(a.raw_probability+b.raw_probability)/2)
            ma,mb,me=macro(pa,oracle),macro(pb,oracle),macro(pe,oracle)
            rows.append(dict(mechanism=mech,generator_seed=s,model=m,single=(ma+mb)/2,ens2=me))
D=pd.DataFrame(rows);D.to_csv('docs/paper/extension-v3/comparator-seed-ensemble.csv',index=False)
x=D.merge(P,on=['mechanism','generator_seed','model'])
print('max |single-csv|',(x.single-x.roc_auc).abs().max(),len(x))
