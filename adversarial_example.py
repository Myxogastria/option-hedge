from deep_hedging import *

model = keras.models.load_model('dh_model.h5')

input_list, target_payoff = next(train_batches)
last_price = input_list[-1]

hedged_payoff = model.predict(input_list)

hedging_error = hedged_payoff - target_payoff

n_pickup = 30

# うまくいかなかった時系列
worst_index = np.argsort(hedging_error, axis=0)[:n_pickup].reshape(-1, )
pd.DataFrame({'last':last_price[worst_index], 'err':hedging_error[worst_index].reshape((-1, ))})
pd.DataFrame({'last':last_price[worst_index], 'err':hedging_error[worst_index].reshape((-1, ))})

input_array = np.stack(input_list)
worst_ts = input_array[:, worst_index]

plt.figure()
plt.plot(worst_ts)
plt.show(block=False)

# うまくいった時系列
best_index = np.argsort(hedging_error, axis=0)[-n_pickup:].reshape(-1, )
pd.DataFrame({'last':last_price[best_index], 'err':hedging_error[best_index].reshape((-1, ))})
pd.DataFrame({'last':last_price[best_index], 'err':hedging_error[best_index].reshape((-1, ))})

input_array = np.stack(input_list)
best_ts = input_array[:, best_index]

plt.figure()
plt.plot(best_ts)
plt.show(block=False)

# 正確な時系列
precise_index = np.argsort(np.abs(hedging_error), axis=0)[:n_pickup].reshape(-1, )
pd.DataFrame({'last':last_price[precise_index], 'err':hedging_error[precise_index].reshape((-1, ))})
pd.DataFrame({'last':last_price[precise_index], 'err':hedging_error[precise_index].reshape((-1, ))})

input_array = np.stack(input_list)
precise_ts = input_array[:, precise_index]

plt.figure()
plt.plot(precise_ts)
plt.show(block=False)


# ヘッジ誤差が大きい時系列とヘッジ誤差が小さい時系列とでは傾向が異なる
col_alpha = 0.1
plt.figure()
plt.plot(worst_ts, color=(1, 0, 0, col_alpha))
plt.plot(best_ts, color=(0, 1, 0, col_alpha))
plt.plot(precise_ts, color=(0, 0, 1, col_alpha))
plt.plot(worst_ts.mean(axis=1), color=(1, 0, 0, 1))
plt.plot(best_ts.mean(axis=1), color=(0, 1, 0, 1))
plt.plot(precise_ts.mean(axis=1), color=(0, 0, 1, 1))
plt.show(block=False)
