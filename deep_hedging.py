import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, Multiply, Subtract, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil

plt.close('all')

S_init = 100
u = 0.1
d = -0.05
p = 0.6

r = 0.01
K = 100

S_0 = Input(shape=(1, ), name='S_0') # price of assets
# W_0 = Dense(1, kernel_initializer=keras.initializers.Constant(value=0.054), use_bias=False, name='W_0')(S_0) # valuation of portfolio
# Vs_0 = Dense(1, kernel_initializer=keras.initializers.Constant(value=0.614), use_bias=False, name='Vs_0')(S_0) # portfolio of stock by value
W_0 = Dense(1, use_bias=False, name='W_0')(S_0) # valuation of portfolio
Vs_0 = Dense(1, use_bias=False, name='Vs_0')(S_0) # portfolio of stock by value
Vr_0 = Subtract(name='Vr_0')([W_0, Vs_0]) # portfolio of risk-free asset by value

S_1 = Input(shape=(1, ), name='S_1')
Rs_1 = Lambda(lambda x: x[0]/x[1], name='Rs_1')([S_1, S_0]) # return of stock
Ws_1 = Multiply(name='Ws_1')([Rs_1, Vs_0]) # valuation of stock after 1 term
Wr_1 = (1+r)*Vr_0 # valuation of risk-free asset after 1 term
W_1 = Add(name='W_1')([Ws_1, Wr_1])
Vs_1 = Dense(1, use_bias=True, name='Vs_1')(W_1)
# Vs_1 = Dense(1, kernel_initializer=keras.initializers.Constant(value=9), 
#     bias_initializer=keras.initializers.Constant(value=11), 
#     use_bias=True, name='Vs_1')(W_1)
Vr_1 = Subtract(name='Vr_1')([W_1, Vs_1])

S_2 = Input(shape=(1, ), name='S_2')
Rs_2 = Lambda(lambda x: x[0]/x[1], name='Rs_2')([S_2, S_1]) # return of stock
Ws_2 = Multiply(name='Ws_2')([Rs_2, Vs_1]) # valuation of stock after 2 term
Wr_2 = (1+r)*Vr_1 # valuation of risk-free asset after 2 term
W_2 = Add(name='W_2')([Ws_2, Wr_2])

model = Model(inputs=[S_0, S_1, S_2], outputs=W_2)

# optimizer = keras.optimizers.RMSprop(lr=0.01)
optimizer = keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss='mean_absolute_error')

model_v = Model(inputs=[S_0, S_1, S_2], outputs=[W_0, Vs_0, Vr_0, Rs_1, Ws_1, Wr_1, W_1, Vs_1, Vr_1, Rs_2, Ws_2, Wr_2, W_2])
model_option_price = Model(inputs=[S_0, S_1, S_2], outputs=W_0)

plot_model(model, show_shapes=True, to_file='model.png')

N = 100
input0 = S_init*np.ones((N, ))
input1 = (1+d+(u-d)*np.random.randint(2, size=(N, )))*input0
input2 = (1+d+(u-d)*np.random.randint(2, size=(N, )))*input1
input_list = [input0, input1, input2]

target_payoff = np.maximum(input2-K, 0).reshape((N, 1))

log_dir = '/tmp/log'
shutil.rmtree(log_dir, ignore_errors=True)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(input_list, target_payoff, batch_size=100, epochs=20000, callbacks=[tensorboard_callback])

pd.DataFrame({'loss': history.history['loss']}).plot()
plt.yscale('log')
plt.grid()

hedged_payoff = model.predict(input_list)
plt.figure()
plt.scatter(input2, hedged_payoff)
plt.grid()

option_price = model_option_price.predict(input_list)
plt.figure()
plt.hist(option_price)

[input0[:2], input1[:2], input2[:2]]
v = model_v.predict([input0[:2], input1[:2], input2[:2]])

plt.show(block=False)



