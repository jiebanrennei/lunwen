import scipy.sparse as sp
feat_a = sp.load_npz("../a_feat.npz").astype("float32")
print(feat_a)