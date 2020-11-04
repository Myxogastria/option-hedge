from deep_hedging import *

model = keras.models.load_model('dh_model.h5')

input_list, target_payoff = next(train_batches)
last_price = input_list[-1]
input_array = np.stack(input_list)

hedged_payoff = model.predict(input_list)

hedging_error = hedged_payoff - target_payoff

n_pickup = 30

plt.close('all')

# うまくいかなかった時系列
worst_index = np.argsort(hedging_error, axis=0)[:n_pickup].reshape(-1, )
pd.DataFrame({'last':last_price[worst_index], 'err':hedging_error[worst_index].reshape((-1, ))})

worst_ts = input_array[:, worst_index]

plt.figure()
plt.plot(worst_ts)
plt.show(block=False)

# うまくいった時系列
best_index = np.argsort(hedging_error, axis=0)[-n_pickup:].reshape(-1, )
pd.DataFrame({'last':last_price[best_index], 'err':hedging_error[best_index].reshape((-1, ))})

best_ts = input_array[:, best_index]

plt.figure()
plt.plot(best_ts)
plt.show(block=False)

# 正確な時系列
precise_index = np.argsort(np.abs(hedging_error), axis=0)[:n_pickup].reshape(-1, )
pd.DataFrame({'last':last_price[precise_index], 'err':hedging_error[precise_index].reshape((-1, ))})

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


hedged_payoff_list = [hedged_payoff]
for j in [1, -1]:
    logret = np.diff(np.log(input_array), axis=0)
    logret_sort = np.sort(logret, axis=0)[::j, :]
    input_sort = np.zeros(input_array.shape)
    input_sort[1:, :] = np.cumsum(logret_sort, axis=0)
    input_sort = S_init*np.exp(input_sort)

    plt.figure()
    plt.plot(input_sort[:, :10])
    plt.plot(input_array[:, :10])
    plt.show(block=False)


    hedged_payoff_sort = model.predict([input_sort[i, :] for i in range(input_sort.shape[0])])
    hedged_payoff_list.append(hedged_payoff_sort)

    hedging_error_sort = hedged_payoff_sort - target_payoff

    # うまくいかなかった時系列
    pd.DataFrame({'last':last_price[worst_index], 'err':hedging_error[worst_index].reshape((-1, )), 'err_sort':hedging_error_sort[worst_index].reshape((-1, ))})

    worst_ts_sort = input_sort[:, worst_index]

    plt.figure()
    plt.plot(worst_ts_sort)
    plt.show(block=False)

    # うまくいった時系列
    pd.DataFrame({'last':last_price[best_index], 'err':hedging_error[best_index].reshape((-1, )), 'err_sort':hedging_error_sort[best_index].reshape((-1, ))})

    best_ts_sort = input_sort[:, best_index]

    plt.figure()
    plt.plot(best_ts_sort)
    plt.show(block=False)

    # 正確な時系列
    pd.DataFrame({'last':last_price[precise_index], 'err':hedging_error[precise_index].reshape((-1, )), 'err_sort':hedging_error_sort[precise_index].reshape((-1, ))})

    precise_ts_sort = input_sort[:, precise_index]

    plt.figure()
    plt.plot(precise_ts_sort)
    plt.show(block=False)


plt.figure()

# 入れ替えなし
plt.scatter(last_price, hedged_payoff_list[0], color=(1, 0, 0, col_alpha))

# 下へ往って来い
# ヘッジしすぎて戻りきれないパターンが見られる
plt.scatter(last_price, hedged_payoff_list[1], color=(0, 1, 0, col_alpha))

# 上へ往って来い
# 利確して目標ペイオフを上回る
plt.scatter(last_price, hedged_payoff_list[2], color=(0, 0, 1, col_alpha))
plt.plot([min(last_price), strike, max(last_price)], [0, 0, max(last_price)-strike])

plt.grid()
plt.show(block=False)
