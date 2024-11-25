import pandas as pd

def update_dataset_info(filepath):
    df = pd.read_excel(filepath)
    columns = df.columns.tolist()
    unique_values = {col: df[col].dropna().unique().tolist() for col in columns}
    return {"columns": columns, "unique_values": unique_values}
