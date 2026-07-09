import pandas as pd
import os

csv_path = 'output_updated.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f'Total rows: {len(df)}')
    print(f'Columns: {list(df.columns)}')
    print(f'\nDate range:')
    print(f'  First: {df["Datetime"].iloc[0]}')
    print(f'  Last: {df["Datetime"].iloc[-1]}')
    print(f'\nLast 5 rows:')
    print(df.tail())
else:
    print(f'File not found: {csv_path}')





