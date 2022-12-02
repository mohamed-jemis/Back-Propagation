from tkinter import Tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import matplotlib.pyplot as plt


def preprocess():
    penguins_df = pd.read_csv('penguins.csv')

    # changing data types from object to category
    penguins_df["gender"] = penguins_df["gender"].astype('category')
    penguins_df["species"] = penguins_df["species"].astype('category')

    # encoding categorical data
    enc = OrdinalEncoder()
    penguins_df["gender_cat"] = enc.fit_transform(penguins_df[["gender"]])
    penguins_df["species_cat"] = enc.fit_transform(penguins_df[["species"]])

    # dropping unused columns
    penguins_df.drop("gender", inplace=True, axis=1)
    # penguins_df.drop("species", inplace=True, axis=1)

    # filling gender na
    penguins_df = penguins_df.apply(lambda x: x.fillna(x.value_counts().index[0]))

    # Scaling numerical features
    scaler = MinMaxScaler()
    penguins_df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]] = scaler.fit_transform(
        penguins_df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])

    penguins_df_groups = penguins_df.groupby("species", group_keys=False)
    groups = penguins_df_groups.groups.keys()

    # Splitting data to train and test
    train_data, test_data = [], []
    #
    # train_data = np.empty(shape=[0,])

    for group in groups:
        curr_group = np.array(penguins_df_groups.get_group(group))
        for i in range(50):
            if i < 30:
                train_data.append(curr_group[i, 1:])
            else:
                test_data.append(curr_group[i, 1:])

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    # print(train_data.shape)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    X_train = train_data[:, :-1]
    Y_train = train_data[:, -1]

    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1]
    return X_train, Y_train, X_test, Y_test


def init_weights(features, neurons, layers, classes):
    weights_in = np.random.rand(neurons, features)
    bias = np.random.rand(layers+1, neurons)
    bias_out = np.random.rand(classes, 1)
    weights_of_hidden = np.random.rand(layers, neurons, neurons)
    weights_out = np.random.rand(classes, neurons)

    weights = []

    weights.append(weights_in)
    weights.append(weights_of_hidden)
    weights.append(weights_out)

    np.append(bias, bias_out)

    return weights, bias


def sigmoid(X):
    X = X.astype(float)
    return 1/1+np.exp(-X)


def forward_step(weights, bias, X, activation):
    print(weights.shape, ' ' , 'Wshape')
    print(X.shape, ' ' , 'Xshape')
    print(bias.shape, ' ' , 'Bshape')
    Z = weights.dot(X.T) + bias
    # print(Z.shape)
    if activation == 1:
        return sigmoid(Z), Z
    else:
        return np.tanh(Z), Z


def forward_propagation(weights, bias, x, activation, layers):
    A, Z = forward_step(weights[0], bias[0], x, activation)
    # print(X.shape())
    A_s, Z_s = [], []
    A_prev = A

    A_s.append(A_prev)
    Z_s.append(Z)

    for i in range(1, layers):
        A_prev = A
        A, Z = forward_step(weights[i], bias[i], A_prev, activation)
        A_s.append(A)
        Z_s.append(Z)

    y, Z = forward_step(weights[layers], bias[layers-1], A, activation)
    Z_s.append(Z)

    Z_s = np.array(Z_s)
    A_s = np.array(A_s)

    return y, A_s, Z_s


def activ_derv(X, activation):
    if activation == 1:
        return sigmoid(X) * (1 - sigmoid(X))
    else:
        return 1 - np.tanh(X)**2


def back_propagation(weights, t, y, Z_s, activation, layers):
    print("in back")
    print(y.shape)
    error_out = (t - y) * activ_derv(Z_s[layers], activation)
    error_h = []

    for i in range(1, layers):
        error = activ_derv(Z_s[i], activation) * np.dot(weights, error_h)
        error_h.append(error)

    error_h = np.array(error_h)

    return error_out, error_h


def update_weights(weights, bias, error_out, error_h, lr, x, layers):
    dwh = lr * error_h * x
    dwy = lr * error_out
    dbh = lr * error_h

    weights[layers] -= dwy
    weights[:layers] -= dwh
    bias[:] -= dbh
    return weights, bias


def train_weight(weights, bias, X, t, activation, layers, epochs, lr):
    # if weights not init do it
    for i in range(epochs):
        for k in range(X.shape[0]):
            y, A_s, Z_s = forward_propagation(weights, bias, X[k], activation, layers)
            error_out, error_h = back_propagation(weights, t, y, Z_s, activation, layers)
            weights, bias = update_weights(weights, bias, error_out, error_h, lr, X[k])

    return weights, bias


def test(weights, bias, X_test, Y_test, activation, layers, classes):
    y, A_s, Z_s = forward_propagation(weights, bias, X_test, activation, layers)
    confusion_matrix = np.zeros((classes, classes))
    correct = 0

    for i in range(Y_test.shape[0]):
        max_indx = np.argmax(y, axis=1)

        # y[np.argmax(y, axis=1)] = 1
        # y[~max_indx] = 0

        if max_indx + 1 == Y_test:
            confusion_matrix[max_indx, max_indx] += 1
        else:
            confusion_matrix[max_indx, Y_test - 1] += 1

    for i in range(classes):
        correct += confusion_matrix[i, i]

    accuracy = correct * 100 / Y_test.shape[0]

    return accuracy, confusion_matrix


def neural_network(neurons, layers, classes, activation, epochs, lr):
    X_train, Y_train, X_test, Y_test = preprocess()
    features = X_train.shape[1]

    weights, bias = init_weights(features, neurons, layers, classes)
    weights, bias = train_weight(weights, bias, X_train, Y_train, activation, layers, epochs, lr)
    accuracy, confusion_matrix = test(weights, bias, X_test, Y_test, activation, layers, classes)

    return accuracy, confusion_matrix


########## GUI
# initializing window
root = Tk()

# window size and title
root.geometry("360x270")
root.title("Penguins")


# initializing labels of NN info
l1 = Label(root, text="Number of hidden layers")
l2 = Label(root, text="No. of neurons in each layer")

# placing labels of NN info
l1.place(x=20, y=15)
l2.place(x=200, y=15)

# initializing NN info textboxes
no_of_layers_txt = Entry(root)
no_of_neurons_txt = Entry(root)

# placing NN info textboxes
no_of_layers_txt.place(x=20, y=35)
no_of_neurons_txt.place(x=200, y=35)

# initializing labels of textboxes
l_eta = Label(root, text="Learning rate")
l_m = Label(root, text="Number of epochs")

# placing labels of textboxes
l_eta.place(x=20, y=65)
l_m.place(x=200, y=65)

# initializing textboxes
eta_txt = Entry(root)
m_txt = Entry(root)

# placing textboxes
eta_txt.place(x=20, y=85)
m_txt.place(x=200, y=85)

# creating bias checkbox
b_var = IntVar()
bias_ck = Checkbutton(root, text="Add bias", variable=b_var)
bias_ck.place(x=140, y=195)

# activation fn radio button
ac_fn_indx = IntVar()
sigmoid_rb = Radiobutton(root, text="Sigmoid activation function", variable=ac_fn_indx, value=1)
tan_rb = Radiobutton(root, text="Hyperbolic Tangent activation function", variable=ac_fn_indx, value=2)

l_ac = Label(root, text="Choose activation function")
l_ac.place(x=20, y=115)

sigmoid_rb.place(x=20, y=135)
tan_rb.place(x=20, y=155)

# Submitting function


def submit_clk():
    eta = float(eta_txt.get())
    m = int(m_txt.get())
    no_of_layers = int(no_of_layers_txt.get())
    no_of_neurons = int(no_of_neurons_txt.get())
    b = np.zeros((1, 1))
    bb = b_var.get()
    activation = ac_fn_indx.get()

    if bb == 1:
        b = np.random.rand(1, 1)

    accuracy, confusion_matrix = neural_network(no_of_neurons, no_of_layers, 3, activation, m, eta)
    messagebox.showinfo("Result", "The accuracy is " + str(accuracy))


# initalizing and placing submit button
submit = Button(root, text="Submit", width=20, command=submit_clk)
submit.place(x=110, y=220)

root.mainloop()