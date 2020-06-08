import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, Multiply, Subtract, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil

plt.close('all')

S_init = 100
u = 0.1
d = -0.05
p = 0.6

r = 0.01
K = 100

class Vr_0_Layer(keras.layers.Layer):
  def __init__(self, **kwargs):
    super(Vr_0_Layer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    self.vars = self.add_weight(
      shape=(2,),
      trainable=True,
      name='W0_Vs0'
    )
    super(Vr_0_Layer, self).build(input_shape)
  
  def get_config(self):
    cfg = super(Vr_0_Layer, self).get_config()
    return cfg
  
  def call(self, null_inputs):
  # def call(self, s_0):
    # y = self.vars * s_0  # = W0_Vs0
    # w_0, vs_0 = y[:, 0], y[:, 1]
    y = self.vars  # = W0_Vs0
    w_0, vs_0 = y[0], y[1]
    return w_0 - vs_0, w_0, vs_0


class Vr_1_Layer(keras.layers.Layer):
  def __init__(self, **kwargs):
    super(Vr_1_Layer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    self.vars = self.add_weight(
      shape=(1,),
      trainable=True,
      name='Vs1'
    )
    super(Vr_1_Layer, self).build(input_shape)
  
  def get_config(self):
    cfg = super(Vr_1_Layer, self).get_config()
    return cfg
  
  def call(self, s_0, s_1, vs_0, vr_0):
    rs_1 = s_1 / s_0
    ws_1 = rs_1 * vs_0
    wr_1 = (1. + r) * vr_0
    w_1 = ws_1 + wr_1
    vs_1 = w_1 * self.vars
    vr_1 = w_1 - vs_1
    return rs_1, ws_1, wr_1, w_1, vs_1, vr_1
  
  # def compute_output_shape(self, input_shape):
  #   return input_shape[0], input_shape[0], input_shape[3], input_shape[0]

S_0 = Input(shape=(1, ), name='S_0') # price of assets
Vr_0, W_0, Vs_0 = Vr_0_Layer()(S_0) # portfolio of risk-free asset by value

S_1 = Input(shape=(1, ), name='S_1')
Rs_1, Ws_1, Wr_1, W_1, Vs_1, Vr_1 = Vr_1_Layer()(S_0, S_1, Vs_0, Vr_0)

# model_test = Model(inputs=[S_0, S_1], outputs=Ws_1)

# N = 100
# input0 = S_init*np.ones((N, ))
# input1 = (1+d+(u-d)*np.random.randint(2, size=(N, )))*input0
# input2 = (1+d+(u-d)*np.random.randint(2, size=(N, )))*input1
# input_list = [input0, input1, input2]

# model_test.predict([input0, input1])

S_2 = Input(shape=(1, ), name='S_2')
Rs_2, Ws_2, Wr_2, W_2, Vs_2, Vr_2 = Vr_1_Layer()(S_1, S_2, Vs_1, Vr_1)

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
model.predict([input0[:2], input1[:2], input2[:2]])

plt.show(block=False)



