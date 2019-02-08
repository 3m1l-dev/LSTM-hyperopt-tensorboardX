"""
Keras CuDNNLSTM hyper-parameter optimization using hyperopt.
"""
import os
import time
import random
import multiprocessing

import matplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve

from tensorboardX import SummaryWriter

from keras.layers import Lambda, Input, Embedding, Dense, concatenate
from keras.layers import Dropout, SpatialDropout1D, CuDNNLSTM, GaussianNoise
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

from hyperopt import fmin, tpe, hp, anneal, Trials

matplotlib.use('agg')
plt.close('all')

# Seed and debug mode option

NEW_SEED = 2018
DEBUG_MODE = False

# Params

EMBED_SIZE = 300 # how big is each word vector
MAX_FEATURES = 95000 # how many unique words to use (i.e num rows in embedding vector)
MAXLEN = 70 # max number of words in a question to use
STAT_FEAT_SIZE = 1 # we have added statistical features to the data
BATCH_SIZE = 512 # how many samples to process at once

N_SPLITS = 5
N_EPOCHS = 5

# Load data

X_TRAIN = np.load("X.dat")
Y_TRAIN = np.load("y.dat")
EMBEDDING_MATRIX = np.load("embedding_matrix.dat")

def seed_keras(seed=NEW_SEED):
    """
    Function for setting a seed for reproduceability
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def threshold_search(y_true, y_proba, plot=False):
    """
    Function to obtain highest F1 and best threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001)
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    if plot:
        plt.plot(thresholds, F, '-b')
        plt.plot([best_th], [best_score], '*r')
        plt.show()
    search_result = {'threshold': best_th, 'f1': best_score}
    return search_result

def get_run_name_from_params(model_params):
    """
    The parameters tuned in this model are the following:

    - lstm layer size : lstm_size
    - embedding dropout : emb_dropout
    - post dense layer dropout : dense_dropout
    - final dropout before activation : final_dropout
    - gaussian noise on/off after loading embedding : gaussian_1
    - gaussian noise amount after loading embedding: g_noise_1
    - gaussian noise on/off on lstm outputs : gaussian_2
    - gaussian noise on lstm outputs: g_noise_2

    We round parameters such as dropout and gaussian noise
    to shorten folder and file names and keep them cleaner.

    """

    perem = model_params.copy()

    perem['emb_dropout'] = round(perem['emb_dropout'], 4)
    perem['dense_dropout'] = round(perem['dense_dropout'], 4)
    perem['final_dropout'] = round(perem['final_dropout'], 4)
    perem['g_noise_1'] = round(perem['g_noise_1'], 4)
    perem['g_noise_2'] = round(perem['g_noise_2'], 4)

    """

    If gaussian noise is used, we only need to store the noise
    amount in the run name, to conserve space once again.

    """

    if perem['gaussian_1'] == 0:
        del perem['g_noise_1']

    if perem['gaussian_2'] == 0:
        del perem['g_noise_2']

    del perem['gaussian_1']
    del perem['gaussian_2']

    run_name = '_'.join('{}_{}'.format(key, val) for key, val in perem.items())
    if run_name == '':
        run_name = "default_run"
    run_name = './runs/'+run_name
    return run_name

def f1(y_true, y_pred):
    """
    F1 metric as we have a highly unbalanced dataset.

    Source:
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

    """
    def recall(y_true, y_pred):
        """
        Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def new_model(lstm_size,
              emb_dropout,
              dense_dropout,
              final_dropout,
              gaussian_1,
              gaussian_2,
              g_noise_1,
              g_noise_2,
              embedding_matrix,
              feat_size=STAT_FEAT_SIZE,
              maxlen=MAXLEN):

    """
    Here we build our single bidirectional CuDNNLSTM layered
    network to detect insincere questions. The input is sliced
    to sequential and statistical data, sequential data being
    run through the recurrent layers and statistical being
    concatenated before the final dense and activation layers.

    """

    inp = Input(shape=(maxlen+feat_size, ))

    x_seq = Lambda(lambda x: x[:, :maxlen], output_shape=(maxlen,))(inp)
    x_stat = Lambda(lambda x: x[:, maxlen:], output_shape=(feat_size,))(inp)

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(x_seq)
    if gaussian_1 == 1:
        x = GaussianNoise(g_noise_1)(x)
    x = SpatialDropout1D(emb_dropout)(x)

    """
    We save the forward and backward hidden states that are outputted as f_h and b_h
    """
    x, f_h, _, b_h, _ = Bidirectional(CuDNNLSTM(lstm_size,
                                                kernel_initializer=glorot_uniform(seed=seed_nb),
                                                return_sequences=True,
                                                return_state=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([f_h, b_h, avg_pool, max_pool])

    if gaussian_2 == 1:
        conc = GaussianNoise(g_noise_2)(conc)

    conc = Dropout(dense_dropout)(conc)
    conc = Dense(64, activation="relu")(conc)
    conc = concatenate([conc, x_stat])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(final_dropout)(conc)
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1])
    return model


def ensembler(param,
              x_train,
              y_train,
              embedding_matrix,
              n_splits=N_SPLITS,
              n_epochs=N_EPOCHS):
    """
    This is the main function that trains our model using K-fold,
    and ensembles the predictions of each fold to calcualte a
    final local F1 score.
    """

    # Setting seed
    seed_keras()

    model_params = {}
    model_params['lstm_size'] = param['lstm_size'] # RNN layer hidden size
    model_params['emb_dropout'] = param['emb_dropout'] # Embedding dropout
    model_params['dense_dropout'] = param['dense_dropout'] # Dense layer dropout
    model_params['final_dropout'] = param['final_dropout'] # Final dropout before out
    model_params['gaussian_1'] = param['gaussian_1'] # Gaussian 1 on/off
    model_params['gaussian_2'] = param['gaussian_2'] # Gaussian 2 on/off
    model_params['g_noise_1'] = param['g_noise_1'] # Gaussian noise 1
    model_params['g_noise_2'] = param['g_noise_2'] # Gaussian noise 2

    # Storing losses and rocaucs per epoch

    a_val_losses = np.zeros((n_splits, n_epochs))
    t_rocaucs = np.zeros((n_splits, 1))

    # Defining a tensorboardX writer

    writer = SummaryWriter(get_run_name_from_params(model_params))

    print("Model params: " + get_run_name_from_params(model_params))

    """

    Train meta will be our predictions on whole train set. This will,
    help in choosing the optimal threshold for the final predictions.

    """

    train_meta = np.zeros(y_train.shape)

    splits = list(StratifiedKFold(n_splits=n_splits,
                                  shuffle=True,
                                  random_state=NEW_SEED).split(x_train, y_train))

    for idx, (train_idx, valid_idx) in enumerate(splits):
        print("Fold: " + str(idx))
        X_train_fold = x_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = x_train[valid_idx]
        y_val_fold = y_train[valid_idx]

        model = new_model(
            param['lstm_size'],
            param['emb_dropout'],
            param['dense_dropout'],
            param['final_dropout'],
            param['gaussian_1'],
            param['gaussian_2'],
            param['g_noise_1'],
            param['g_noise_2'],
            embedding_matrix)

        start_time = time.time()

        hist = model.fit(
            X_train_fold,
            y_train_fold,
            batch_size=512,
            epochs=n_epochs,
            validation_data=(X_val_fold, y_val_fold),
            verbose=0)

        pred_val_y = model.predict([X_valf], batch_size=1024, verbose=0)

        train_meta[valid_idx] = pred_val_y.reshape(-1)

        search_result = threshold_search(y_val_fold, pred_val_y)

        best_score = metrics.f1_score(y_val_fold, (pred_val_y > search_result['threshold']).astype(int))


        """

        Here we store the validation losses per epoch so we can
        send them to the tensorboardX writer and observe the
        output.

        """
        avg_val_losses = []

        for e in range(n_epochs):
            avg_val_losses.append(hist.history['val_loss'][e])

        end_time = time.time()
        elapsed_time = round((end_time-start_time), 1)

        print("Epoch: ", e+1, " Val F1: {:.4f}".format(best_score) + " Time: " + elapsed_time)


        a_val_losses[idx] = avg_val_losses
        t_rocaucs[idx] = roc_auc_score(y_valf, pred_val_y)

    final_score = threshold_search(y_train, train_meta)

    a_val_losses = np.average(a_val_losses, axis=0).tolist()
    final_roc_auc = np.average(t_rocaucs, axis=0)[0]
    final_thresh = final_score['threshold']
    final_f1 = final_score['f1']

    # Writing to tensorboardX

    for e in range(n_epochs):
        writer.add_scalar('avg_val_loss', a_val_losses[e], e)

    writer.add_scalar('final_roc_auc', final_roc_auc, 0)
    writer.add_scalar('final_thresh', final_thresh, 0)
    writer.add_scalar('final_f1', final_f1, 0)

    print("final_f1: "+str(final_f1))

    # In this case we want to optimise F1.

    return final_f1


def local_cv(params, x_train=X_TRAIN, y_train=Y_TRAIN, embedding_matrix=embedding_matrix):

    """
    This function gets a set of variable parameters in "param",
    sets param datatypes, loads data and runs the ensembler function.

    """

    param = {'lstm_size': int(params['lstm_size']),
             'emb_dropout': params['emb_dropout'],
             'dense_dropout': params['dense_dropout'],
             'final_dropout': params['final_dropout'],
             'gaussian_1': params['gaussian_1'],
             'gaussian_2': params['gaussian_2'],
             'g_noise_1': params['g_noise_1'],
             'g_noise_2': params['g_noise_2']}

    # always call this before training for deterministic results
    seed_keras()

    final_f1_score = ensembler(param, x_train, y_train, embedding_matrix)

    # IMPORTANT: Reset memory between testing models.
    K.clear_session()

    """

    IMPORTANT NOTE: Hyperopt optimizes a function by minimizing a
    value you tell it to. In this case, we want to MAXIMIZE the F1
    score, and the easiest way to make hyperopt do this is to add
    a minus sign infront of the F1 in order for hyperopt to minimize
    correctly.

    In other optimization frameworks such as Optuna, you are given
    the choice to maximize or minimize which makes things much easier
    to understand, however in hyperopt we resort to this for now.

    """

    return -final_f1_score


if __name__ == "__main__":

    # Load data.

    if DEBUG_MODE == True:
        X_TRAIN = X_TRAIN[:50000]
        Y_TRAIN = Y_TRAIN[:50000]
        N_SPLITS = 2
        N_EPOCHS = 2

    # Parameter space for hyperopt to explore

    SPACE = {'rs': hp.quniform('rs', 30, 60, 4),
             'ed': hp.uniform('ed', 0., 0.5),
             'dd': hp.uniform('dd', 0., 0.5),
             'od': hp.uniform('od', 0., 0.5),
             'g1': hp.choice('g1', [0, 1]),
             'g2': hp.choice('g2', [0, 1]),
             'gn1': hp.uniform('gn1', 0., 0.3),
             'gn2': hp.uniform('gn2', 0., 0.3)}



    # Trials will contain logging information
    TRIALS = Trials()

    BEST = fmin(fn=local_cv, # function to optimize
                space=SPACE,
                algo=tpe.suggest, # optimization algorithm Tree Parzen Estimator
                max_evals=20, # maximum number of iterations
                trials=TRIALS, # logging
                rstate=np.random.RandomState(4950) # fixing random state for reproducibility
               )

    print("Best Local CV {:.3f} params {}".format(local_cv(BEST), BEST))

    """

    Best model parameters are outputted to a text file in the same directory, and also printed above.

    """

    with open("Output_stats.txt", 'a') as out:
        out.write("Best performing model chosen hyper-parameters: {}".format(best) + '\n')
