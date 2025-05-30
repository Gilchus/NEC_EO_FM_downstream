import pandas as pd
import numpy as np
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def load_embeddings(file_path):
    df = pd.read_csv(file_path)
    return df


def average_embeddings(df1, df2, weight1=0.5, weight2=0.5, embedding_dim=1024):
    assert weight1 + weight2 == 1.0, "Weights must sum to 1"
    merged = pd.merge(df1, df2, on='ID', suffixes=('_1', '_2'))
    emb_columns = [f'{i}' for i in range(embedding_dim)]
    averaged = merged[["ID"]].copy()
    for i in range(embedding_dim):
        averaged[str(i)] = weight1 * merged[f'{i}_1'] + weight2 * merged[f'{i}_2']
    return averaged


def concatenate_embeddings(df1, df2):
    merged = pd.merge(df1, df2, on='ID', suffixes=('_1', '_2'))
    concatenated = merged[["ID"]].copy()
    for i in range(1024):
        concatenated[f'{i}'] = merged[f'{i}_1']
    for i in range(1024):
        concatenated[f'{1024 + i}'] = merged[f'{i}_2']
    return concatenated


# Check if this can be applied
def pca_fuse_embeddings(dfs, embedding_dim, id_col='ID', n_components=None):
    """
    Merge multiple embedding DataFrames and fuse via PCA.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Each DataFrame must have an 'ID' column plus embedding columns named
        '0', '1', ..., f'{embedding_dim-1}'.
    embedding_dim : int
        Dimensionality D of each individual embedding vector.
    id_col : str, default 'ID'
        Name of the identifier column common to all dfs.
    n_components : int or None, default None
        Number of principal components to keep.  If None, defaults to embedding_dim.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns [id_col, '0', '1', …, f'{n_components-1}'] where each
        row is the fused embedding for that ID.
    """
    if n_components is None:
        n_components = embedding_dim

    # 1) Merge all on ID
    merged = reduce(lambda a, b: pd.merge(a, b, on=id_col, suffixes=(False, False)), dfs)

    # 2) Build the concatenated feature matrix
    #    We assume each df contributed columns '0','1',…,'D-1' which now exist
    #    in merged as duplicated names; pandas auto‐renames to .1, .2, etc.
    #    To avoid that, we re-suffix explicitly before merging:
    #    (See note below if you want automatic suffixing.)

    # --- if you instead pre‐suffix each df, you'd do:
    # merged = dfs[0].rename(columns={str(i): f'{i}_1' for i in range(embedding_dim)})
    # for k, df in enumerate(dfs[1:], start=2):
    #     merged = merged.merge(
    #         df.rename(columns={str(i): f'{i}_{k}' for i in range(embedding_dim)}),
    #         on=id_col
    #     )

    feature_cols = []
    for df_idx in range(1, len(dfs) + 1):
        feature_cols += [f'{i}_{df_idx}' for i in range(embedding_dim)]

    X = merged[feature_cols].values  # shape (n_samples, len(dfs)*embedding_dim)

    # 3) PCA reduction back down to n_components
    pca = PCA(n_components=n_components)
    fused = pca.fit_transform(X)      # shape (n_samples, n_components)

    out = pd.DataFrame(
        fused,
        columns=[str(i) for i in range(n_components)]
    )
    out.insert(0, id_col, merged[id_col].values)
    return out


def save_embeddings(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine two embedding CSVs")
    parser.add_argument("file1", help="Path to first CSV file")
    parser.add_argument("file2", help="Path to second CSV file")
    parser.add_argument("method", choices=["average", "concat", "linear"], help="Merging method")
    parser.add_argument("output", help="Output CSV file path")
    parser.add_argument("--weight1", type=float, default=0.5, help="Weight for first embedding (only for average)")
    parser.add_argument("--weight2", type=float, default=0.5, help="Weight for second embedding (only for average)")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="Embedding dimension of csv files")
    args = parser.parse_args()

    df1 = load_embeddings(args.file1)
    df2 = load_embeddings(args.file2)

    if args.method == "average":
        result = average_embeddings(df1, df2, args.weight1, args.weight2)
    elif args.method == "concat":
        result = concatenate_embeddings(df1, df2)

    save_embeddings(result, args.output)
