import pandas as pd
import sys
sys.path.insert(0, '.')
import config as cfg

df = pd.read_csv(cfg.DATA_PATH)
print('Dataset shape:', df.shape)
print('\nClass distribution:')
for col in cfg.TARGET_COLS:
    counts = df[col].value_counts()
    total = len(df)
    print(f'\n{col}:')
    print(f'  0 (negative): {counts.get(0, 0)} ({counts.get(0, 0)/total*100:.2f}%)')
    print(f'  1 (positive): {counts.get(1, 0)} ({counts.get(1, 0)/total*100:.2f}%)')
    print(f'  Imbalance ratio: {counts.get(0, 0) / max(counts.get(1, 1), 1):.2f}:1')
