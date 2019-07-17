import numpy as np


def print_daleness(w_inp, w_rec):
    if np.all(w_inp >= 0):
        print('dale: all inputs weights are nonnegative')
    else:
        print('not dale: some input weights are negative')
    w_rec_sign = np.sign(w_rec)
    if np.all(np.all(w_rec_sign == w_rec_sign[0, :], 0)):
        print('dale: all recurrent weight columns have consistent sign')
    else:
        print('not dale: some recurrent weight columns have mixed signs')


def daleness(w_inp, w_rec, w_out):
    w_inp_mean = np.mean(np.fabs(w_inp))
    w_inp_neg = w_inp[w_inp < 0]
    return np.mean(np.fabs(w_inp_neg)) / w_inp_mean


def relu(x):
    return x * (x > 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip_0to1(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x
