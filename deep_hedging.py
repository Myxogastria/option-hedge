import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, Multiply, Subtract, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shutil

plt.close('all')

n_terms = 30
n_weight = 10

# epochs:3000, loss:0.6, price:2.3
epochs = 3000

S_init = 100
strike = 100

u = 0.1
d = -0.05
u = (1+u)**(1/n_terms)-1
d = (1+d)**(1/n_terms)-1
p = 0.6

r = 0.01

def binom_iter(input_size):
    def input_generator():
        while True:
            input_list = [S_init*np.ones((input_size, ))]
            for i in range(n_terms):
                input_list.append((1+d+(u-d)*np.random.randint(2, size=(input_size, )))*input_list[-1])

            target_payoff = np.maximum(input_list[-1]-strike, 0).reshape((input_size, 1))
            yield input_list, target_payoff
    return 1, input_generator()

class GeometricBrownianMotion:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def get_path(self, time, spot):
        dt = time[1:] - time[:-1]
        S = np.insert((self.mu - 1/2*self.sigma**2)*dt \
            + self.sigma*np.random.normal(0, np.sqrt(dt)), 0, np.log(spot)).cumsum()
        
        return np.exp(S)
        # return pd.Series(np.exp(S), time)
    
    def get_generator(self, time, spot, input_size):
        def input_generator():
            while True:
                input_np = np.stack([self.get_path(time, spot) for i in range(input_size)]).T
                target_payoff = np.maximum(input_np[-1]-strike, 0).reshape((input_size, 1))
                yield [input_np[i] for i in range(input_np.shape[0])], target_payoff
        return 1, input_generator()

time = np.arange(0, n_terms+1)/360

mu = 0.07
sigma = 0.20

stockModel = GeometricBrownianMotion(mu, sigma)


N = 1000

# r = (1+r)**(1/n_terms)-1
# train_steps, train_batches = binom_iter(N)

r = np.exp(r/360)-1
train_steps, train_batches = stockModel.get_generator(time, S_init, N)



# W_0, Vs_0を入力なしの変数にしたいが，
# そうする方法がよくわからないので
# 常に同じ値を入力するS_0を入力としている．
S_list = [Input(shape=(1, ), name='S_0')] # price of assets
W_list = [Dense(1, use_bias=False, name='W_0')(S_list[-1])] # valuation of portfolio
Vs_list = [Dense(1, use_bias=False, name='Vs_0')(S_list[-1])] # portfolio of stock by value
Vr_list = [Subtract(name='Vr_0')([W_list[-1], Vs_list[-1]])] # portfolio of risk-free asset by value

for i in range(1, n_terms+1):
    S_list.append(Input(shape=(1, ), name='S_{}'.format(i)))
    Rs_i = Lambda(lambda x: x[1]/x[0], name='Rs_{}'.format(i))(S_list[-2:]) # return of stock
    Ws_i = Multiply(name='Ws_{}'.format(i))([Rs_i, Vs_list[-1]]) # valuation of stock after 1 term
    Wr_i = (1+r)*Vr_list[-1] # valuation of risk-free asset after 1 term
    W_list.append(Add(name='W_{}'.format(i))([Ws_i, Wr_i]))
    if i < n_terms:
        dense_i = Dense(n_weight, activation='relu', use_bias=True, name='Vs_{}_0'.format(i))(W_list[-1])
        for j in range(1, min(i, 3)):
            dense_i = Dense(n_weight, activation='relu', use_bias=True, name='Vs_{}_{}'.format(i, j))(dense_i)
        Vs_list.append(Dense(1, use_bias=True, name='Vs_{}_{}'.format(i, i))(dense_i))
        Vr_list.append(Subtract(name='Vr_{}'.format(i))([W_list[-1], Vs_list[-1]]))

model = Model(inputs=S_list, outputs=W_list[-1])

# optimizer = keras.optimizers.RMSprop(lr=0.01)
optimizer = keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss='mean_absolute_error')

model_option_price = Model(inputs=S_list, outputs=W_list[0])

# plot_model(model, show_shapes=True, to_file='model.png')

if __name__ == '__main__':
    model.summary()
    
    
    log_dir = '/tmp/log'
    shutil.rmtree(log_dir, ignore_errors=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # history = model.fit(input_list, target_payoff, batch_size=N, epochs=20000, callbacks=[tensorboard_callback])
    history = model.fit(train_batches, steps_per_epoch=train_steps, epochs=epochs, callbacks=[tensorboard_callback])
    model.save('dh_model.h5')
    
    pd.DataFrame({'loss': history.history['loss']}).plot()
    plt.yscale('log')
    plt.grid()
    
    input_list, _ = next(train_batches)
    
    hedged_payoff = model.predict(input_list)
    plt.figure()
    plt.scatter(input_list[-1], hedged_payoff)
    plt.grid()
    
    option_price = model_option_price.predict(input_list)
    print(option_price[0][0])
    
    plt.show(block=False)
    


