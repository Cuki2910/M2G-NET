"""Recompute primary numbers independently, then emit traceable paper tables."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config as cfg
from scripts.generate_paper_simulation import sha256


def independent_ap(targets, probabilities):
    selected=pd.DataFrame({'target':targets,'probability':probabilities})
    if not len(selected) or not selected.target.sum():
        return np.nan
    grouped=selected.groupby('probability',sort=True).target.agg(['sum','count']).sort_index(ascending=False)
    precision=grouped['sum'].cumsum()/grouped['count'].cumsum()
    return float((precision*grouped['sum']/selected.target.sum()).sum())


def main():
    folder=Path('docs/paper')
    plan=json.loads((folder/'run-plan.json').read_text())
    manifest=json.loads((folder/'aggregation-manifest.json').read_text())
    for name,expected in manifest['outputs'].items():
        assert sha256(folder/name)==expected,name
    results=pd.read_csv(folder/'results.csv').set_index(['unit_id','task'])
    summary=pd.read_csv(folder/'results-summary.csv')
    deltas=pd.read_csv(folder/'paired-summary.csv')
    budgets=pd.read_csv(folder/'fit-budgets.csv')
    errors=[]
    assert len(results)==750 and len(plan['units'])==150
    for unit in plan['units']:
        root=Path(unit['output'])/f"random_seed_{unit['training_seed']}"/unit['model']
        predicted=pd.read_csv(root/'predictions.csv')
        oracle=pd.read_csv(Path(unit['data_directory'])/'oracle.csv').set_index('simulation_row_id')
        task_scores=[]
        for task in cfg.TASK_NAMES:
            selected=predicted.loc[predicted.task.eq(task)].set_index('row_id')
            truth=oracle.loc[selected.index]
            eligible=truth[task+'_eligible'].eq(1)
            targets=truth.loc[eligible,task+'_latent'].to_numpy()
            probabilities=selected.loc[eligible,'raw_probability'].to_numpy()
            positives=int(targets.sum()); negatives=len(targets)-positives
            ranks=pd.Series(probabilities).rank(method='average').to_numpy()
            auc=(ranks[targets==1].sum()-positives*(positives+1)/2)/(positives*negatives)
            brier=float(np.mean((probabilities-targets)**2))
            ap=independent_ap(targets,probabilities)
            recorded=results.loc[(unit['id'],task)]
            assert recorded.n_eligible==len(targets) and recorded.n_positive==positives
            assert recorded.n_observed==int(selected.observed.sum())
            errors.extend([abs(auc-recorded.roc_auc),abs(brier-recorded.brier),abs(ap-recorded.average_precision)])
            task_scores.append(recorded[['roc_auc','average_precision','brier']].to_numpy(dtype=float))
        np.testing.assert_allclose(np.mean(task_scores,axis=0),results.loc[(unit['id'],'macro'),['roc_auc','average_precision','brier']].to_numpy(dtype=float),rtol=0,atol=1e-14)
    assert max(errors)<1e-12
    flat=results.reset_index()
    for _,row in summary.iterrows():
        selected=flat.loc[flat.mechanism.eq(row.mechanism)&flat.model.eq(row.model)&flat.task.eq(row.task)].sort_values(['generator_seed','training_seed'])
        assert len(selected)==6
        for metric in ['roc_auc','average_precision','brier']:
            values=selected[metric].to_numpy().reshape(3,2)
            replicate=values.mean(axis=1)
            expected={'mean':replicate.mean(),'sd':replicate.std(ddof=1),'min':replicate.min(),'max':replicate.max(),
                      'mcse':replicate.std(ddof=1)/np.sqrt(3),'mean_training_sd':values.std(axis=1,ddof=1).mean()}
            for suffix,value in expected.items():
                assert abs(row[metric+'_'+suffix]-value)<1e-12
    raw_deltas=pd.read_csv(folder/'paired-deltas.csv')
    for _,row in raw_deltas.iterrows():
        selected=flat.loc[flat.mechanism.eq(row.mechanism)&flat.generator_seed.eq(row.generator_seed)&flat.training_seed.eq(row.training_seed)&flat.task.eq(row.task)].set_index('model')
        for metric in ['roc_auc','average_precision','brier']:
            assert abs(selected.loc['m2g_net',metric]-selected.loc[row.model,metric]-row[metric])<1e-12
    assert budgets.loc[budgets.model.isin(['logistic_regression','catboost']),'candidate_fits'].eq(12).all()
    assert budgets.loc[~budgets.model.isin(['logistic_regression','catboost']),'candidate_fits'].eq(1).all()
    text=['# Audited tables','', 'All values use raw predictions and eligible oracle outcomes. Average two training seeds within each of three independent generator replicates; dataset SD uses ddof=1. AUC/AP higher is better; Brier lower is better. No significance claim. Task denominators are not pooled across endpoints.','']
    def table(caption,headers,rows):
        text.extend([caption,'','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        text.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
        text.append('')
    counts=[]
    for (mechanism,task),group in flat.loc[~flat.task.eq('macro')].groupby(['mechanism','task'],sort=False):
        counts.append([mechanism,task]+[f'{int(group[column].min())}-{int(group[column].max())}' for column in ['n_eligible','n_positive','n_observed']])
    table('Table 1. Eligible test denominators across three datasets (600 outer test rows each; repeated model/training-seed counts are not additional observations).',['Mechanism','Task','Eligible','Positive','Observed'],counts)
    macro=summary.loc[summary.task.eq('macro')]
    table('Table 2. Macro endpoints: mean and sample SD over three seed-averaged generator replicates. Training SD is the mean within-replicate SD over two training seeds.',['Mechanism','Procedure','AUC mean','Dataset SD','Training SD','AP mean','Brier mean'],[[row.mechanism,row.model,*[f'{row[column]:.6f}' for column in ['roc_auc_mean','roc_auc_sd','roc_auc_mean_training_sd','average_precision_mean','brier_mean']]] for _,row in macro.iterrows()])
    task_rows=[]
    for _,row in macro.iterrows():
        group=summary.loc[summary.mechanism.eq(row.mechanism)&summary.model.eq(row.model)].set_index('task')
        task_rows.append([row.mechanism,row.model]+[f'{group.loc[task,"roc_auc_mean"]:.6f}' for task in cfg.TASK_NAMES])
    table('Table 3. Per-task mean eligible AUC; negative comparisons are retained.',['Mechanism','Procedure','Red-light','Turn-signal','Helmet','Phone'],task_rows)
    table('Table 4. Paired macro AUC differences, full minus comparator; positive favors full. SD, min/max and descriptive MCSE use three replicate-level, training-seed-averaged deltas.',['Mechanism','Comparator','Mean delta','SD','Minimum','Maximum','MCSE'],[[row.mechanism,row.model,*[f'{row[column]:.6f}' for column in ['roc_auc_mean','roc_auc_sd','roc_auc_min','roc_auc_max','roc_auc_mcse']]] for _,row in deltas.loc[deltas.task.eq('macro')].iterrows()])
    budget_rows=[]
    for model,group in budgets.groupby('model',sort=False):
        parameter='not comparable' if group.trainable_model_parameters.isna().all() else str(int(group.trainable_model_parameters.iloc[0]))
        budget_rows.append([model,len(group),parameter,int(group.candidate_fits.min()),f'{int(group.n_train.min())}-{int(group.n_train.max())}',f'{int(group.n_validation.min())}-{int(group.n_validation.max())}',f'{group.wall_seconds.sum():.2f}'])
    table('Table 5. Actual execution budgets. Trainable neural counts include dormant instantiated modules, exclude fixed temperatures; counts do not establish active-capacity matching. Classical candidate fits are summed over four tasks.',['Procedure','Units','Trainable model parameters','Candidate fits/unit','Train rows','Validation rows','Fit-process seconds'],budget_rows)
    text.extend(['Provenance: `aggregation-manifest.json` hashes every table input; `run-plan.json` identifies all units and source hashes. Detailed unit/task endpoints, paired deltas, ranges, MCSE and split counts remain in the corresponding CSVs. No development pilot or historical score enters these tables.',''])
    target=folder/'tables.md'
    rendered='\n'.join(text)
    if target.exists():
        assert target.read_text()==rendered,'Audited table content changed; preserve and review before replacement'
    else:
        target.write_text(rendered)
    audit={'status':'passed','units':150,'task_auc_brier_ap_recomputations':600,'macro_checks':150,
           'summary_rows_checked':len(summary),'paired_task_seed_deltas_checked':len(raw_deltas),
           'maximum_absolute_independent_error':max(errors),'table_sha256':sha256(target),
           'script_sha256':sha256(__file__),'aggregation_manifest_sha256':sha256(folder/'aggregation-manifest.json')}
    (folder/'number-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
