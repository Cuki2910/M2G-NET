"""Isolated eligible-population scoring after every frozen fit is accepted."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from scripts.run_paper_unit import read, verify, verify_sources
from scripts.generate_paper_simulation import sha256


METRICS = ("roc_auc", "average_precision", "brier")


def metrics(targets, probabilities):
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    if len(targets) != len(probabilities) or not np.isin(targets, [0, 1]).all():
        raise ValueError("Invalid scoring labels")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid prediction probabilities")
    if not len(targets):
        return {name: np.nan for name in METRICS}
    return {"roc_auc": roc_auc_score(targets, probabilities) if len(np.unique(targets)) == 2 else np.nan,
            "average_precision": average_precision_score(targets, probabilities) if targets.sum() else np.nan,
            "brier": brier_score_loss(targets, probabilities)}


def score_predictions(predictions, oracle, unit, test_rows):
    if set(predictions.task.unique()) != set(cfg.TASK_NAMES):
        raise ValueError("Missing or extra task predictions")
    if oracle.simulation_row_id.duplicated().any():
        raise ValueError("Duplicate oracle row IDs")
    rows = []
    for task in cfg.TASK_NAMES:
        selected = predictions.loc[predictions.task.eq(task)].copy()
        if selected.row_id.duplicated().any() or set(selected.row_id) != set(test_rows):
            raise ValueError("Test row membership mismatch")
        joined = selected.merge(oracle, left_on="row_id", right_on="simulation_row_id", validate="one_to_one", how="left", indicator=True)
        if not joined['_merge'].eq('both').all() or not joined.site_id.eq(joined[cfg.SITE_ID_COL]).all():
            raise ValueError("Oracle join/site mismatch")
        eligible = joined[task + "_eligible"].eq(1)
        mask = joined[cfg.TASK_MASK_COLS[task]]
        if not joined.observed.eq(mask).all() or (mask > eligible.astype(int)).any():
            raise ValueError("Oracle/prediction observation mismatch")
        observed = mask.eq(1)
        if not joined.loc[observed, 'target'].eq(joined.loc[observed, task + '_latent']).all():
            raise ValueError("Observed outcome mismatch")
        score = metrics(joined.loc[eligible, task + '_latent'], joined.loc[eligible, 'raw_probability'])
        diagnostic = metrics(joined.loc[observed, task + '_latent'], joined.loc[observed, 'raw_probability'])
        rows.append({"unit_id": unit['id'], "mechanism": unit['mechanism'], "generator_seed": unit['generator_seed'],
                     "training_seed": unit['training_seed'], "model": unit['model'], "task": task,
                     "n_test": len(joined), "n_eligible": int(eligible.sum()), "n_positive": int(joined.loc[eligible, task+'_latent'].sum()),
                     "n_observed": int(observed.sum()), **score, **{'observed_'+name: value for name,value in diagnostic.items()}})
    macro = {key: rows[0][key] for key in ('unit_id','mechanism','generator_seed','training_seed','model','n_test')}
    macro.update(task='macro', n_eligible=np.nan, n_positive=np.nan, n_observed=np.nan)
    for name in (*METRICS, *('observed_'+metric for metric in METRICS)):
        values = np.array([row[name] for row in rows])
        macro[name] = float(values.mean()) if np.isfinite(values).all() else np.nan
    return rows + [macro]


def summarize(frame):
    rows = []
    for keys, group in frame.groupby(['mechanism','model','task'], sort=False):
        row = dict(zip(('mechanism','model','task'), keys))
        replicates = []
        seed_sds = []
        for seed, sample in group.groupby('generator_seed'):
            if len(sample) != 2 or set(sample.training_seed) != {42,101}:
                raise ValueError("Missing or duplicate training seeds")
            replicates.append(sample[list(METRICS)].mean(skipna=False))
            seed_sds.append(sample[list(METRICS)].std(ddof=1,skipna=False))
        if len(replicates) != 3:
            raise ValueError("Missing generator replicates")
        values = pd.DataFrame(replicates)
        within = pd.DataFrame(seed_sds)
        row.update(n_generator_replicates=3,training_seeds_per_replicate=2)
        for metric in METRICS:
            selected = values[metric]
            row.update({metric+'_mean': selected.mean(skipna=False), metric+'_sd': selected.std(ddof=1,skipna=False),
                        metric+'_min': selected.min(skipna=False), metric+'_max': selected.max(skipna=False),
                        metric+'_mcse': selected.std(ddof=1,skipna=False)/np.sqrt(3),
                        metric+'_mean_training_sd': within[metric].mean(skipna=False)})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    plan = read('docs/paper/run-plan.json')
    state = read('docs/PAPER_TASK_STATE.json')
    if state['tasks']['P13']['status'] != 'done' or len(state['run_units']) != len(plan['units']) or any(unit['status']!='done' for unit in state['run_units']):
        raise ValueError('All frozen fits must complete before oracle scoring')
    verify_sources(plan)
    all_rows, provenance, budgets, counts = [], [], [], []
    trainable_counts = {}
    for unit in plan['units']:
        manifest, evaluation, model_root, generated = verify(unit)
        accepted = read(Path(unit['output'])/'paper-unit.json')
        if accepted['plan_sha256'] != sha256('docs/paper/run-plan.json'):
            raise ValueError('Acceptance plan changed')
        for filename, expected in accepted['acceptance_sha256'].items():
            if sha256(filename) != expected:
                raise ValueError('Acceptance artifact changed')
        data_dir = Path(unit['data_directory'])
        oracle = pd.read_csv(data_dir/'oracle.csv')
        predictions = pd.read_csv(model_root/'predictions.csv')
        if not predictions.model.eq(unit['model']).all() or not predictions.seed.eq(unit['training_seed']).all() or not predictions.fold.eq('random').all():
            raise ValueError('Prediction identity mismatch')
        all_rows.extend(score_predictions(predictions,oracle,unit,evaluation['split_rows']['test']))
        selections = evaluation['selection']
        candidate_count = sum(len(value['candidates']) for value in selections) if unit['model'] in ('catboost','logistic_regression') else 1
        model_parameters = accepted['parameter_counts']['model']
        cache_key = (unit['model'], model_parameters)
        if cache_key not in trainable_counts:
            if (model_root/'model.pt').exists():
                from src.checkpoint import load_model_bundle
                bundle = load_model_bundle(model_root/'model.pt', data_path=data_dir/'data.csv')
                trainable_counts[cache_key] = sum(value.numel() for value in bundle['model'].parameters() if value.requires_grad)
            else:
                trainable_counts[cache_key] = model_parameters
        budgets.append({'unit_id':unit['id'],'mechanism':unit['mechanism'],'model':unit['model'],
                        'wall_seconds':accepted['wall_seconds'],'instantiated_model_parameters':accepted['parameter_counts']['model'],
                        'trainable_model_parameters':trainable_counts[cache_key],
                        'loss_parameters':accepted['parameter_counts']['loss'],'candidate_fits':candidate_count,
                        **{'n_'+key: value for key,value in accepted['split_counts'].items()}})
        data = pd.read_csv(data_dir/'data.csv')
        for split, ids in evaluation['split_rows'].items():
            selected = data.set_index('simulation_row_id').loc[ids]
            for task in cfg.TASK_NAMES:
                mask = selected[cfg.TASK_MASK_COLS[task]].eq(1)
                counts.append({'unit_id':unit['id'],'split':split,'task':task,'rows':len(selected),
                               'eligible':int(selected[task+'_eligible'].sum()),'observed':int(mask.sum()),
                               'observed_positive':int(selected.loc[mask,task].sum())})
        provenance.append({'unit_id':unit['id'],'acceptance_sha256':sha256(Path(unit['output'])/'paper-unit.json'),
                           'predictions_sha256':sha256(model_root/'predictions.csv'),'oracle_sha256':sha256(data_dir/'oracle.csv')})
    results = pd.DataFrame(all_rows)
    if len(results)!=len(plan['units'])*5 or results.duplicated(['unit_id','task']).any():
        raise ValueError('Incomplete scoring matrix')
    summary = summarize(results)
    contrasts = []
    full=results.loc[results.model.eq('m2g_net')]
    for model in results.model.unique():
        if model=='m2g_net':
            continue
        paired=full.merge(results.loc[results.model.eq(model)],on=['mechanism','generator_seed','training_seed','task'],suffixes=('_full','_comparison'),validate='one_to_one')
        for _,row in paired.iterrows():
            contrasts.append({key:row[key] for key in ('mechanism','generator_seed','training_seed','task')} |
                             {'model':model} | {metric:row[metric+'_full']-row[metric+'_comparison'] for metric in METRICS})
    delta=pd.DataFrame(contrasts)
    folder=Path('docs/paper')
    outputs={'results.csv':results,'results-summary.csv':summary,'paired-deltas.csv':delta,
             'paired-summary.csv':summarize(delta),'fit-budgets.csv':pd.DataFrame(budgets),'split-counts.csv':pd.DataFrame(counts)}
    for name,frame in outputs.items():
        if (folder/name).exists():
            raise FileExistsError('Preserve prior aggregation: '+name)
    for name,frame in outputs.items():
        frame.to_csv(folder/name,index=False,float_format='%.17g')
    audit={'plan_sha256':sha256('docs/paper/run-plan.json'),'scorer_sha256':sha256(__file__),'units':len(plan['units']),
           'undefined_eligible_auc':int(results.roc_auc.isna().sum()),'inputs':provenance,
           'outputs':{name:sha256(folder/name) for name in outputs}}
    (folder/'aggregation-manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('PASS: scored',len(results),'unit/task endpoints;',audit['undefined_eligible_auc'],'undefined AUCs')


if __name__=='__main__':
    main()
