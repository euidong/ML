from sklearn.datasets._samples_generator import make_blobs
import pandas as pd
from util.main import generate_filename_to_parent_directory

X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)

dfX = pd.DataFrame(X, columns=['X1', 'X2'])
dfY = pd.DataFrame(Y, columns=['Y'])

df = pd.merge(dfX, dfY, left_index=True, right_index=True)

df.to_csv(
    generate_filename_to_parent_directory(__file__, ext='.csv'), index=False
)
