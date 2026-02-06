def run(df):
    df["new_col"] = df["value"] * 2
    return df