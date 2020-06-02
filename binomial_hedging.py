import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, Multiply, Subtract, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

import numpy as np

S_init = 100
u = 0.1
d = -0.05
p = 0.6

r = 0.01
K = 100

S_0 = Input(shape=(1, ), name='S_0') # stock
x_0 = Dense(1, use_bias=False, name='x_0')(S_0) # amount of stock
V_0 = Multiply()([S_0, x_0]) # value of portfolio

S_1 = Input(shape=(1, ), name='S_1')
x_1 = Dense(1, name='x_1')(keras.layers.concatenate([S_1, x_0]))
V_1 = Multiply()([S_1, x_0])

S_2 = Input(shape=(1, ), name='S_2')
x_2 = Dense(1, name='x_2')(keras.layers.concatenate([S_2, x_1]))
V_2 = Add()([Multiply()([S_2, x_1]), (1+r)*Subtract()([V_1, Multiply()([S_1, x_1])])])

option_payoff = Lambda(lambda x: (keras.backend.abs(x-K)+x-K)/2)(S_2)

hedge_error = Subtract()([V_2, option_payoff])

model = Model(inputs=[S_0, S_1, S_2], outputs=[hedge_error, V_2])

def null_loss(x_true, x_pred):
    return 0*x_true

model.compile(optimizer='rmsprop', loss=['mean_absolute_error', null_loss])

plot_model(model, show_shapes=True, to_file='model.png')

N = 100
input0 = S_init*np.ones((N, ))
input1 = (1+d+(u-d)*np.random.randint(2, size=(N, )))*input0
input2 = (1+d+(u-d)*np.random.randint(2, size=(N, )))*input1
input_list = [input0, input1, input2]

model.fit(input_list, [np.zeros((N, )), np.zeros((N, ))], epochs=300)

result_error, result_payoff = model.predict(input_list)

plt.figure()
plt.scatter(input2, result_payoff)
plt.show()



