
import numpy as np
from sklearn.cluster import KMeans

def build_vocab(desc_list, clusters=200, max_samples=30000):
    all_desc = np.vstack([d for d in desc_list if d is not None])
    if len(all_desc) > max_samples:
        idx = np.random.choice(len(all_desc), max_samples, replace=False)
        all_desc = all_desc[idx]
    km = KMeans(n_clusters=clusters, n_init=10, random_state=42)
    km.fit(all_desc)
    return km
