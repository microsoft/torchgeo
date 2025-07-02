import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------------------------------------------
# 1) Input CSVs (each has "id" + embedding columns)
# -------------------------------------------------------------------
csv1_path = "./submissions/embeddings_clip_convnext_xxlarge_256_ftregress_e22.csv"  # 256->128
csv2_path = "./submissions/embeddings_clip_VITH_256_E25.csv"              # 256->192
csv3_path = "./submissions/embedding_vit_base_finetune_epoch0030.csv"          # 768->192
csv4_path = "./submissions/georsclip_submission_512.csv"            # already compressed to 512

# -------------------------------------------------------------------
# 2) Read them into memory
# -------------------------------------------------------------------
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)
df3 = pd.read_csv(csv3_path)
df4 = pd.read_csv(csv4_path)

# -------------------------------------------------------------------
# 3) Merge all on 'id' to get a single set of common IDs and consistent row order
# -------------------------------------------------------------------
ids_df = (
    df1[['id']]
    .merge(df2[['id']], on='id', how='inner')
    .merge(df3[['id']], on='id', how='inner')
    .merge(df4[['id']], on='id', how='inner')
)
print("Number of rows (common IDs):", len(ids_df))

# -------------------------------------------------------------------
# 4) CSV1: columns [1:257] (256 columns), SVD => 128
# -------------------------------------------------------------------
cols1 = df1.columns[1:257]  # embedding columns from index 1..256
df1_aligned = ids_df.merge(df1[['id'] + list(cols1)], on='id', how='left')
X1 = df1_aligned.drop(columns=['id']).values

svd1 = TruncatedSVD(n_components=128, random_state=42)
X1_reduced = svd1.fit_transform(X1)
print(
    f"CSV1: shape in={X1.shape}, shape out={X1_reduced.shape}, "
    f"expl_var={svd1.explained_variance_ratio_.sum():.4f}"
)

# -------------------------------------------------------------------
# 5) CSV2: columns [1:257] (256 columns), SVD => 192
# -------------------------------------------------------------------
cols2 = df2.columns[1:257]
df2_aligned = ids_df.merge(df2[['id'] + list(cols2)], on='id', how='left')
X2 = df2_aligned.drop(columns=['id']).values

svd2 = TruncatedSVD(n_components=192, random_state=42)
X2_reduced = svd2.fit_transform(X2)
print(
    f"CSV2: shape in={X2.shape}, shape out={X2_reduced.shape}, "
    f"expl_var={svd2.explained_variance_ratio_.sum():.4f}"
)

# -------------------------------------------------------------------
# 6) CSV3: columns [1:769] (768 columns), SVD => 192
# -------------------------------------------------------------------
cols3 = df3.columns[1:769]
df3_aligned = ids_df.merge(df3[['id'] + list(cols3)], on='id', how='left')
X3 = df3_aligned.drop(columns=['id']).values

svd3 = TruncatedSVD(n_components=192, random_state=42)
X3_reduced = svd3.fit_transform(X3)
print(
    f"CSV3: shape in={X3.shape}, shape out={X3_reduced.shape}, "
    f"expl_var={svd3.explained_variance_ratio_.sum():.4f}"
)

# -------------------------------------------------------------------
# 7) CSV4: already pre‐compressed to 512 → skip SVD
# -------------------------------------------------------------------
cols4 = df4.columns[1:513]  # embedding columns from index 1..512
df4_aligned = ids_df.merge(df4[['id'] + list(cols4)], on='id', how='left')
X4_reduced = df4_aligned.drop(columns=['id']).values
print(
    f"CSV4: shape in={X4_reduced.shape}, shape out={X4_reduced.shape} "
    "(no additional SVD performed)"
)

# -------------------------------------------------------------------
# 8) Combine all reduced arrays → total = 128 + 192 + 192 + 512 = 1024
# -------------------------------------------------------------------
X_final = np.concatenate([X1_reduced, X2_reduced, X3_reduced, X4_reduced], axis=1)
print("Final shape (X_final):", X_final.shape)

# -------------------------------------------------------------------
# 9) (Optional) Evaluate with KMeans + silhouette
# -------------------------------------------------------------------
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X_final)
sil_score = silhouette_score(X_final, labels)
print(f"Silhouette score: {sil_score:.4f}")

# -------------------------------------------------------------------
# 10) Save final result (id + 1024 columns)
# -------------------------------------------------------------------
final_df = pd.DataFrame(X_final, columns=[f"svd_{i}" for i in range(X_final.shape[1])])
final_df.insert(0, 'id', ids_df['id'])
final_df.to_csv("./submissions/cxxft_vith_dinobfttest_4season512.csv", index=False)
print("✅ Saved final 1024‐D embeddings.")

