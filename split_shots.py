from train import *

x, y, z = preprocess(data_path)
x, x_test, z, z_test = train_test_split(x, z, test_size = test_fraction, random = False)

trainshots = np.array(list(set(z)), dtype=np.int)
testshots = np.array(list(set(z_test)), dtype=np.int)
d = {'train': trainshots, 'test': testshots}

#np.save('testshots.npy', testshots)
np.save('shot_split.npy', d)
