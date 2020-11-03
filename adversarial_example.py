from deep_hedging import *

model = keras.models.load_model('dh_model.h5')

input_list, target_payoff = next(train_batches)
last_price = input_list[-1]

hedged_payoff = model.predict(input_list)

hedging_error = hedged_payoff - target_payoff

# うまくいかなかった時系列
worst_index = np.argsort(hedging_error, axis=0)[:10].reshape(-1, )
pd.DataFrame({'last':last_price[worst_index], 'err':hedging_error[worst_index].reshape((-1, ))})
pd.DataFrame({'last':last_price[worst_index], 'err':hedging_error[worst_index].reshape((-1, ))})

input_array = np.stack(input_list)
worst_ts = input_array[:, worst_index]

plt.figure()
plt.plot(worst_ts)
plt.show(block=False)

# うまくいった時系列
best_index = np.argsort(hedging_error, axis=0)[-10:].reshape(-1, )
pd.DataFrame({'last':last_price[best_index], 'err':hedging_error[best_index].reshape((-1, ))})
pd.DataFrame({'last':last_price[best_index], 'err':hedging_error[best_index].reshape((-1, ))})

input_array = np.stack(input_list)
best_ts = input_array[:, best_index]

plt.figure()
plt.plot(best_ts)
plt.show(block=False)

plt.figure()
plt.scatter(input_list[-1], hedged_payoff)
plt.grid()
plt.show(block=False)

option_price = model_option_price.predict(input_list)
print(option_price[0][0])
