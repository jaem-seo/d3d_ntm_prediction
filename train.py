import glob, pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

data_path = '/projects/EKOLEMEN/profile_predictor/DATA/profile_data_50ms.pkl'
tm_data_path = '/projects/EKOLEMEN/profile_predictor/DATA/ntm_labels.pkl'
inputs = [
    'bt', 'curr', 'target_density', # Field variables
    'pinj', 'tinj', 'ech', # H/CD variables
    'rmagx_EFIT01', 'a_EFIT01', 'kappa_EFIT01', 'triangularity_top_EFIT01', 'triangularity_bot_EFIT01', # Shape variables
    'betan_EFIT01_prev', 'li_EFIT01_prev', 'qmin_EFIT01_prev' # Previous state
] # Inputs
outputs = [
    'betan_EFIT01', 'li_EFIT01', 'qmin_EFIT01', 'tm'
] # Outputs

n_trial = 5
jump = 1 # down-sampling (1: no downsample)
window = 20
val_fraction = 0.2
test_fraction = 0.1

# Plot setting
lims = [[-0.1, 4.5], [0.0, 4.0], [0.0, 10.0], [-0.1, 1.1]]

def preprocess(data_path):
    try:
        xy = pd.read_csv('preprocessed.csv')
        x = np.zeros([len(xy), window, len(inputs)])
        for i in range(window):
            x[window - 1:, i, :] = xy[inputs].values[i:len(xy) - window + i + 1]
        for i in range(window - 1):
            x[i, window - i - 1:, :] = xy[inputs].values[:i + 1]
        z = xy['shot'].values
        y = xy[outputs].values
        for i in range(window - 1, len(x)):
            if z[i] != z[i - window + 1]:
                for j in range(window):
                    if z[i] != z[i - j]: x[i, window - 1 - j, :] = 0.
        return x.astype(np.float32), y.astype(np.float32), z
    except:
        pass
    
    # Load and combine data
    df = pd.read_pickle(data_path)
    df_tm = pd.read_pickle(tm_data_path)
    shot_list = list(df_tm.keys())
    n_shot = len(shot_list)
    xy = pd.DataFrame(columns = inputs + outputs + ['shot', 'time'])
    for i, shot in enumerate(shot_list):
        try:
            shot_data = df[shot]
            # Get qmin
            shot_data['qmin_EFIT01'] = shot_data['q_EFIT01'].min(axis = 1)
            # Get tm data
            shot_data['tm'] = df_tm[shot]
            # Get previous state
            for o in outputs:
                shot_data[o + '_prev'] = shot_data[o]
                shot_data[o + '_prev'][jump:] = shot_data[o][:-jump]
            shot_data['shot'] = np.ones(len(shot_data['time'])) * shot
            tmp_xy = pd.DataFrame({k: shot_data[k][::jump] for k in inputs + outputs + ['shot', 'time']})
            xy = xy.append(tmp_xy)
        except:
            print('Error:', shot)
            pass

    # Filter NaN and outliers
    xy = xy.dropna()
    xy = xy[xy['curr'] > 0]
    xy = xy[xy['triangularity_top_EFIT01'] > 0]
    xy = xy[xy['triangularity_top_EFIT01'] < 1]
    xy = xy[xy['triangularity_bot_EFIT01'] > 0]
    xy = xy[xy['triangularity_bot_EFIT01'] < 1]
    xy = xy[xy['betan_EFIT01'] > 0]
    xy = xy[xy['qmin_EFIT01'] < 20]

    # Save preprocessed data
    xy.to_csv('preprocessed.csv')

    x = np.zeros([len(xy), window, len(inputs)])
    for i in range(window):
        x[window - 1:, i, :] = xy[inputs].values[i:len(xy) - window + i + 1]
    for i in range(window - 1):
        x[i, window - i - 1:, :] = xy[inputs].values[:i + 1]
    z = xy['shot'].values
    y = xy[outputs].values
    for i in range(window - 1, len(x)):
        if z[i] != z[i - window + 1]:
            for j in range(window):
                if z[i] != z[i - j]: x[i, window - 1 - j, :] = 0.
    return x.astype(np.float32), y.astype(np.float32), z

def train_test_split(x, y, test_size, random=True, random_state=0):
    idx = np.arange(len(x))
    if random:
        np.random.seed(random_state)
        np.random.shuffle(idx)
    n_train = int(len(x) * (1 - test_size))
    return x[idx[:n_train]], x[idx[n_train:]], y[idx[:n_train]], y[idx[n_train:]]

if __name__ == '__main__':
    # Preprocess
    x, y, z = preprocess(data_path)
    x, x_test, y, y_test = train_test_split(x, y, test_size = test_fraction, random = False)

    # Set noise level for preventing overfitting to prev output state
    noise = [0] * x.shape[-1]
    noise[-3:] = 0.25 * y.std(axis=0)[:3]
    
    for seed in range(n_trial):
        # Set random seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = val_fraction, random_state = seed)

        # Build model
        model = keras.models.Sequential()
        model.add(keras.layers.Masking(mask_value = 0., input_shape = x.shape[1:]))
        model.add(keras.layers.GaussianNoise(noise))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LSTM(128, return_sequences=True))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LSTM(64, return_sequences=False))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(y.shape[1], activation='linear'))

        model.summary()

        # Compile model
        model.compile(
            optimizer = 'adam',
            loss = 'mse',
        )
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

        # Train and save model
        hist = model.fit(x_train, y_train, batch_size=int(len(x_train) ** 0.5), epochs=200, callbacks=callbacks, validation_data=(x_val, y_val), verbose=2)
        model.save(f'best_model_{seed}')
        with open(f'training_history_{seed}.pkl', 'wb') as pk:
            pickle.dump(hist.history, pk)
        
        # Plot result
        yy_train = model.predict(x_train)
        yy_val = model.predict(x_val)
        yy_test = model.predict(x_test)

        fig, axs = plt.subplots(3, y.shape[1], figsize = (12, 8))
        for i in range(len(outputs)):
            axs[0, i].scatter(y_train[:, i], yy_train[:, i], s=2, alpha=0.2, label='Train')
            axs[1, i].scatter(y_val[:, i], yy_val[:, i], s=2, alpha=0.2, label='Val')
            axs[2, i].scatter(y_test[:, i], yy_test[:, i], s=2, alpha=0.2, label='Test')
            axs[0, i].set_title(outputs[i])
            for j in range(3):
                lim = lims[i]
                axs[j, i].plot(lim, lim, 'k--')
                axs[j, i].set_xlim(lim)
                axs[j, i].set_ylim(lim)
                axs[j, i].legend()

        #plt.show()
        plt.savefig(f'Regression_{seed}.png')

