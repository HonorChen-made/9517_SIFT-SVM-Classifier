
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_bow(desc_list, km):
    bow = []
    for desc in desc_list:
        if desc is None:
            bow.append(np.zeros(km.n_clusters))
        else:
            hist, _ = np.histogram(km.predict(desc), bins=np.arange(km.n_clusters + 1))
            bow.append(hist)
    return np.array(bow)
