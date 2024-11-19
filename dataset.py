import pandas as pd

splits = {'train': 'data/train-00000-of-00001-ec54fbe500fc3b5c.parquet', 'validation': 'data/validation-00000-of-00001-3cf888b12fff1dd6.parquet'}
df = pd.read_parquet("hf://datasets/lucadiliello/newsqa/" + splits["validation"])

df = df_original[['context', 'question', 'answers']].copy()
df['answers'] = df['answers'].apply(lambda x: x[0] if x else None)

print(df.shape)
