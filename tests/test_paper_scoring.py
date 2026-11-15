import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.summarize_paper_results import metrics, score_predictions, summarize
from scripts.generate_paper_simulation import generate
from scripts.audit_paper_numbers import independent_ap
import config as cfg


class ScoringTest(unittest.TestCase):
    def test_metrics_undefined_and_replication(self):
        self.assertEqual(independent_ap([0,1],[.1,.9]),1)
        self.assertEqual(independent_ap([0,1],[.5,.5]),.5)
        self.assertAlmostEqual(independent_ap([1,0,1],[.8,.8,.2]),7/12)
        self.assertTrue(np.isnan(independent_ap([0,0],[.2,.8])))
        self.assertEqual(metrics([0,1],[.1,.9])['roc_auc'],1)
        self.assertEqual(metrics([0,1],[.5,.5])['roc_auc'],.5)
        self.assertTrue(np.isnan(metrics([1,1],[.2,.8])['roc_auc']))
        self.assertTrue(np.isnan(metrics([],[])['roc_auc']))
        with self.assertRaises(ValueError):
            metrics([0,1],[.1,np.nan])
        rows=[]
        for generator,score in [(1701,.5),(1702,.6),(1703,.7)]:
            for seed,offset in [(42,-.01),(101,.01)]:
                rows.append(dict(mechanism='test',model='test',task='macro',generator_seed=generator,training_seed=seed,
                                 roc_auc=score+offset,average_precision=score+offset,brier=score+offset))
        summary=summarize(pd.DataFrame(rows)).iloc[0]
        self.assertAlmostEqual(summary.roc_auc_mean,.6)
        self.assertAlmostEqual(summary.roc_auc_sd,.1)
        self.assertAlmostEqual(summary.roc_auc_mean_training_sd,np.sqrt(2)*.01)
        with self.assertRaises(ValueError):
            summarize(pd.DataFrame(rows[:-1]))

    def test_join_uses_unobserved_eligible_labels(self):
        data,oracle=generate(11,'partial',120)
        predictions=[]
        for task in cfg.TASK_NAMES:
            predictions.append(pd.DataFrame({'task':task,'row_id':data.simulation_row_id,'site_id':data[cfg.SITE_ID_COL],
                                             'target':data[task].fillna(0),'observed':data[cfg.TASK_MASK_COLS[task]],
                                             'raw_probability':np.full(len(data),.5)}))
        predictions=pd.concat(predictions,ignore_index=True)
        unit={'id':'test','mechanism':'partial','generator_seed':11,'training_seed':42,'model':'test'}
        scored=score_predictions(predictions,oracle,unit,list(range(120)))
        self.assertTrue(any(row['n_eligible']>row['n_observed'] for row in scored[:-1]))
        self.assertTrue(all(row['roc_auc']==.5 for row in scored))
        with self.assertRaises(ValueError):
            score_predictions(pd.concat([predictions,predictions.iloc[:1]]),oracle,unit,list(range(120)))
        bad=oracle.copy();bad[cfg.SITE_ID_COL]=99
        with self.assertRaises(ValueError):
            score_predictions(predictions,bad,unit,list(range(120)))


if __name__=='__main__':
    unittest.main()
