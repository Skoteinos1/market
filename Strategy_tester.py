# Tested 100 times
# chaikin, ema cross, donchian channels, parabolic sar,

# Trading RUSh - Zle: golden cross, aroon, supertrend, opposite(k zlej strategii), connors rsi (horsie ako rsi),

# Vyskusane: macd, rsi, cci, bollinger bands, ichimoku cloud, vortex, alligator, mfi,

# Neskusam: stochastic rsi, bb+rsi, stochastic, macd + stochastic, fractal, beep boop(sam si vytvoril),
# awesome osc., bullish eng., bearish eng., 0lag,

# adx - ukazuje ci je market trending, na odfiltrovanie falosnych signalov

import yfinance as yf
import pandas as pd
import datetime
import time
import talib
import numpy as np
# from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import My_Share_Module as MSM
import _thread
import threading

tickerSymbol = ''
wallet = 1000
bought = False
order = [0, 0]  # ['money', 'volume', price]
order2 = [0]  # ['price']
trading_outcomes = []
percentchange = []
# df = {}
fee = 8.95
price_col_name = 'Adj Close'
price_col_indx = 4
sell_on_exit_points = True
reward = 1.2
risk = 0.9
short_report = False
min_max_rsi = [0, 100]
output_above_profit = 1000
output_above_trades = 60
strategy = ''
which_iter = False
# which_iter = [9, 10, 2]  # zacni tymto
# which_iter = [9, 11, 2]  # zacni tymto
# which_iter = [9, 12, 3]  # zacni tymto

'''
# toto pouzil aby zo stlpca: Date spravil index
df = df.set_index(pd.DatetimeIndex(df['Date'].values))
'''


def get_data(ticker, strt, endd, prd, intr):
    # get data on this ticker
    tickerData = yf.Ticker(ticker)
    prpst = False
    at_adj = False
    if prd:
        df = tickerData.history(period=prd, interval=intr, prepost=prpst, auto_adjust=at_adj)
    else:
        df = tickerData.history(start=strt, end=endd, prepost=prpst, auto_adjust=at_adj, interval=intr)
        prd = str(strt) + ' ' + str(endd)[:10] + ' '
    if intr:
        intr += ' '
    name = tickerSymbol + ' ' + intr + prd + ' ' + str(round(time.time()))
    if prpst:
        name += ' prpst'
    if at_adj:
        name += ' adj'
    MSM.save_pickle(name + '.pkl', df)
    # print(df.to_string())
    with open(name + '.txt', "w") as fo:
        fo.write(df.to_string())
    return


def merge_dfs(df1, df2):
    df1 = MSM.load_pickle(df1)
    print(df1.to_string())
    df2 = MSM.load_pickle(df2)
    print(df2.to_string())

    # test ci treba dat na daco pozor
    merged0 = pd.concat([df1, df2]).drop_duplicates()
    duplicateRowsDF = merged0[merged0.index.duplicated(keep=False)]
    print("\nDuplicates:\n" + duplicateRowsDF.to_string())

    # vyberu spolecne
    common_index = df1.index.intersection(df2.index)
    # z prvni vemu pouze ty ktere nejsou v druhe
    df1x = df1.loc[~df1.index.isin(common_index)]
    merged = pd.concat([df1x, df2])
    return merged


def analyza_testu(vysledky, st_var):
    gains = 0
    ng = 0
    losses = 0
    nl = 0
    total_r_allin = 1
    total_r = 0
    first_day = df.index[0]
    parametre = st_var

    for i in vysledky:
        if i > 0:
            gains += i
            ng += 1
        else:
            losses += i
            nl += 1
        total_r_allin *= (i + 1)
        total_r += i
    total_r_allin = round((total_r_allin - 1) * 100, 2)
    total_r = round(total_r * 100, 2)

    if ng > 0:
        avg_gain = gains / ng
        max_r = max(vysledky)
        max_r = str(round(max_r, 3))
    else:
        avg_gain = 0
        max_r = 'undefined'

    if nl > 0:
        avg_loss = losses / nl
        max_l = min(vysledky)
        max_l = str(round(max_l, 3))
        ratio = -avg_gain / avg_loss
    else:
        avg_loss = 0
        max_l = 'undefined'
        ratio = 'inf'

    if ng > 0 or nl > 0:
        batting_avg = ng / (ng + nl)
    else:
        batting_avg = 0

    itr = ''
    if which_iter:
        itr = parametre.split(' ')
        itr = itr[2]
    end_report = " Uspesnost " + str(round(batting_avg, 3)) + " Trades " + str((ng + nl)) + \
                 " AverageGain " + str(round(avg_gain, 3)) + " AverageLoss " + str(round(avg_loss, 3)) + \
                 " MaxReturn " + max_r + " MaxLoss " + max_l + ' Profit ' + str(total_r_allin)
    if short_report:
        if total_r_allin < 2000:
            print(parametre + ',' + str(total_r_allin))
            return
        else:
            print(parametre + end_report + '%')
            with open(strategy + " " + tickerSymbol + itr + ".csv", "a") as fo:
                # with open(strategy + " " + tickerSymbol + ".csv", "a") as fo:
                fo.write(parametre + end_report + '\n')
    else:
        print("\nResults for " + tickerSymbol + " going back to " + str(first_day) + ", Sample size: " + str(ng + nl) +
              " trades\nStrategy:         ", parametre, "\nBatting Avg:      ", batting_avg, "\nGain/loss ratio:  ",
              ratio,
              "\nAverage Gain:     ", avg_gain, "\nAverage Loss:     ", avg_loss, "\nMax Return:       ", max_r,
              "\nMax Loss:         ", max_l, "\nTotal return over", (ng + nl), "trades:", total_r, "%", '\nAll in:',
              21 * " ",
              total_r_allin, '%')
        with open(strategy + " " + tickerSymbol + ".csv", "a") as fo:
            fo.write(parametre + end_report + '\n')


def buy_order(idx, p=0):
    global bought
    if bought:
        return
    global order
    global order2
    global wallet
    if not p:
        try:
            price = df.iloc[idx, df.columns.get_loc('Open')]
        except:
            return
    else:
        price = p
    price2 = df.iloc[idx-1, df.columns.get_loc(price_col_name)]
    volume = (wallet-fee) // price
    if volume < 1:
        return
    money = round(volume*price + fee, 2, )
    order = [money, volume, price]
    order2 = [price2]
    wallet -= money
    bought = True
    # print('Buyng at', price)


def sell_order(idx, p=0):
    global bought
    if not bought:
        return
    global order
    global wallet
    global trading_outcomes
    if not p:
        try:
            price = df.iloc[idx, df.columns.get_loc('Open')]
        except:
            price = df.iloc[idx-1, df.columns.get_loc('Close')]
    else:
        price = p
    '''if not sell_on_exit_points:
        if order[2]*risk < price < order[2]*reward:
            return
        # elif price > order[2]*reward:
        #     order[2] = price
        #     return'''
    price2 = df.iloc[idx-1, df.columns.get_loc(price_col_name)]
    volume = order[1]
    money = round(volume * price - fee, 2)
    profit = money / order[0] - 1
    order = [0, 0]
    wallet += money
    bought = False
    trading_outcomes.append(profit)
    # if profit < - 0.2:
    #     print(volume, money, round(profit, 4), round(wallet, 2))
    percentchange.append(price2/order2[0]-1)


def trade(io_pts, str_var):
    global wallet
    global trading_outcomes
    global percentchange

    wallet = 1000
    trading_outcomes = []
    percentchange = []

    if sell_on_exit_points:
        for point in io_pts:
            if point[2] == 'buy':
                buy_order(point[0])
            elif point[2] == 'sell':
                sell_order(point[0])
    else:
        order_id = 0
        for c, i in enumerate(df.index, 0):
            if not bought:
                if c == io_pts[order_id][0] and io_pts[order_id][2] == 'buy':
                    buy_order(io_pts[order_id][0], io_pts[order_id][3])
                    order_id += 1
            else:
                if io_pts[order_id][2] == 'sell':
                    if df['High'][i] >= io_pts[order_id][4]:
                        sell_order(io_pts[order_id][0], io_pts[order_id][4])
                        order_id += 1
                    elif df['Low'][i] <= io_pts[order_id][3]:
                        sell_order(io_pts[order_id][0], io_pts[order_id][3])
                        order_id += 1
                    elif c == io_pts[order_id][0]:
                        sell_order(io_pts[order_id][0])
                        order_id += 1
    '''
    # Tato cast kodu by mala byt ak chces robit nejake 2:1 reward to risk test
    if not sell_on_io:
        price = 0
        c = 0
        if bought and not order[2] * risk < price < order[2] * reward:
            sell_order(c + 1)'''

    analyza_testu(trading_outcomes, str_var)
    # analyza_testu(percentchange, str_var)


def strategy_end(eep, rp, sv):
    if not short_report:
        trade(eep, sv)
    elif rp > output_above_profit and len(eep) > output_above_trades:
        trade(eep, sv)


def red_white_blue_strategy(ema1, ema2, ema3, ema4):
    entry_exit_points = []
    rough_profit = 1
    bght = False
    # emasUsed = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]
    emasUsed = [ema1, ema2, ema3, ema4]
    # mins = [0, 0, 0, 0, 0, 0]
    # maxs = [0, 0, 0, 0, 0, 0]
    for ema_x in emasUsed:
        if "Ema_" + str(ema_x) not in df.columns:
            df["Ema_" + str(ema_x)] = round(df.iloc[:, price_col_indx].ewm(span=ema_x, adjust=False).mean(), 2)

    # print(df.tail().to_string())
    # print(df[0:50].to_string())
    for c, i in enumerate(df.index, 0):
        # cmin = min(df['Ema_3'][i], df['Ema_5'][i], df['Ema_8'][i], df['Ema_10'][i], df['Ema_12'][i], df['Ema_15'][i])
        # cmax = min(df['Ema_30'][i], df['Ema_35'][i], df['Ema_40'][i], df['Ema_45'][i], df['Ema_50'][i], df['Ema_60'][i])
        # cmin = min(df['Ema_15'][i],10000)
        # cmax = min(df['Ema_60'][i],10000)
        cmin = min(df['Ema_' + str(ema1)][i], df['Ema_' + str(ema2)][i])
        cmax = min(df['Ema_' + str(ema3)][i], df['Ema_' + str(ema4)][i])
        if cmin > cmax and not bght:
            # foo = [df['Ema_15'][i]]
            # mins[foo.index(min(foo))] += 1
            entry_exit_points.append([c+1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif cmin < cmax and bght:
            # foo = [df['Ema_60'][i]]
            # maxs[foo.index(min(foo))] += 1
            entry_exit_points.append([c+1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False

    if bght:
        entry_exit_points.append([c+1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = 'EMAs: ' + str(emasUsed)
    strategy_end(entry_exit_points, rough_profit-1, strategy_variables)
    # print(mins)
    # print(maxs)
    return rough_profit-1, len(entry_exit_points)


def simple_moving_average(ma=10):
    global df
    bght = False
    entry_exit_points = []
    rough_profit = 1
    sma_str = "Sma_" + str(ma)
    df2 = df
    df2[sma_str] = df2.iloc[:, price_col_indx].rolling(window=ma).mean()
    # zmaze prve riadky ktore nemaju moving average
    # df = df.iloc[ma:]
    for c, i in enumerate(df.index, 0):
        price = df2[price_col_name][i]
        if not bght and price > df2[sma_str][i]:
            entry_exit_points.append([c+1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif bght and price < df2[sma_str][i]:
            entry_exit_points.append([c+1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False

    if bght:
        entry_exit_points.append([c+1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "Simple Moving Average " + str(ma)
    strategy_end(entry_exit_points, rough_profit-1, strategy_variables)
    return rough_profit-1, len(entry_exit_points)


def rsi(days=14, low=30, hi=70):
    def rma(x, n, y0):
        a = (n - 1) / n
        ak = a ** np.arange(len(x) - 1, -1, -1)  # if ak.min() == 0: print('Bude problem')   ak = [x if x != 0 else 5e-324 for x in ak]
        return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a ** np.arange(1, len(x) + 1)]

    # https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe/57037866
    global df
    global min_max_rsi
    n = days
    rough_profit = 1
    bght = False
    entry_exit_points = []

    if 'change' not in df.columns:
        # df['change'] = df2.iloc[:, pci].diff()
        df['change'] = df[price_col_name].diff()
        df['gain'] = df.change.mask(df.change < 0, 0.0)
        df['loss'] = -df.change.mask(df.change > 0, -0.0)

    rsi_str = 'rsi_' + str(days)
    avg_gain_str = 'avg_gain_' + str(days)
    avg_loss_str = 'avg_loss_' + str(days)
    if rsi_str not in df.columns:
        df[rsi_str] = np.nan
        df[avg_gain_str] = np.nan
        df[avg_loss_str] = np.nan
        if 'rsi_' + str(days-1) in df.columns:
            df = df.drop(columns=['rsi_' + str(days-1), 'avg_gain_' + str(days - 1), 'avg_loss_' + str(days - 1)])
        if days < 14:
            for c, i in enumerate(df.index, 0):
                if c < days:
                    continue
                if c == days:
                    gain = 0
                    loss = 0
                    for d in range(days):
                        gain += df.iloc[c - d, df.columns.get_loc('gain')]
                        loss += df.iloc[c - d, df.columns.get_loc('loss')]
                    gain /= days
                    loss /= days
                else:
                    gain = (gain * (days - 1) + df.iloc[c, df.columns.get_loc('gain')]) / days
                    loss = (loss * (days - 1) + df.iloc[c, df.columns.get_loc('loss')]) / days
                if loss == 0:
                    df.iloc[c, df.columns.get_loc(rsi_str)] = 100 - (100 / (1 + gain / 0.0000000001))
                else:
                    df.iloc[c, df.columns.get_loc(rsi_str)] = 100 - (100 / (1 + gain / loss))
        else:
            # Z nejakeho magickeho dovodu to blbne ak su days<13 a len(df)> +-2600
            df[avg_gain_str] = rma(df.gain[n + 1:].to_numpy(), n, np.nansum(df.gain.to_numpy()[:n + 1]) / n)
            df[avg_loss_str] = rma(df.loss[n + 1:].to_numpy(), n, np.nansum(df.loss.to_numpy()[:n + 1]) / n)
            # df2['rs'] = df.avg_gain / df.avg_loss
            # df[rsi_str] = 100 - (100 / (1 + df.rs))
            df[rsi_str] = 100 - (100 / (1 + df[avg_gain_str] / df[avg_loss_str]))
        min_max_rsi = [df[rsi_str].min(), df[rsi_str].max()]
        print(min_max_rsi)
    # print(df[:50].to_string())

    for c, i in enumerate(df.index, 0):
        # price = df2[price_col_name][i]
        if not bght and low > df[rsi_str][i]:
            entry_exit_points.append([c+1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif bght and hi < df[rsi_str][i]:
            entry_exit_points.append([c+1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False

    if bght:
        entry_exit_points.append([c+1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "RSI Days:" + str(days) + ' Buy_S:' + str(low) + ' Sell_S:' + str(hi)
    strategy_end(entry_exit_points, rough_profit-1, strategy_variables)
    return rough_profit-1, len(entry_exit_points)


def connors_rsi(days=14, low=30, hi=70, roc=100):
    # https://www.backtrader.com/recipes/indicators/crsi/crsi/
    # CRSI(3, 2, 100) = [RSI(3) + RSI(Streak, 2) + PercentRank(100)] / 3

    def rma(x, n, y0):
        a = (n - 1) / n
        ak = a ** np.arange(len(x) - 1, -1,
                            -1)  # if ak.min() == 0: print('Bude problem')   ak = [x if x != 0 else 5e-324 for x in ak]
        return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a ** np.arange(1, len(x) + 1)]

    # https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe/57037866
    global df
    global min_max_rsi
    n = days
    rough_profit = 1
    bght = False
    entry_exit_points = []

    if 'change' not in df.columns:
        # df['change'] = df2.iloc[:, pci].diff()
        df['change'] = df[price_col_name].diff()
        df['gain'] = df.change.mask(df.change < 0, 0.0)
        df['loss'] = -df.change.mask(df.change > 0, -0.0)

    rsi_str = 'rsi_' + str(days)
    avg_gain_str = 'avg_gain_' + str(days)
    avg_loss_str = 'avg_loss_' + str(days)
    days -= 1
    if rsi_str not in df.columns:
        df[rsi_str] = np.nan
        df[avg_gain_str] = np.nan
        df[avg_loss_str] = np.nan
        if 'rsi_' + str(days - 1) in df.columns:
            df = df.drop(columns=['rsi_' + str(days), 'avg_gain_' + str(days), 'avg_loss_' + str(days)])
        if days < 14:
            for c, i in enumerate(df.index, 0):
                if c < days:
                    continue
                if c == days:
                    gain = 0
                    loss = 0
                    for d in range(days):
                        gain += df.iloc[c - d, df.columns.get_loc('gain')]
                        loss += df.iloc[c - d, df.columns.get_loc('loss')]
                    gain /= days
                    loss /= days
                else:
                    gain = (gain * (days - 1) + df.iloc[c, df.columns.get_loc('gain')]) / days
                    loss = (loss * (days - 1) + df.iloc[c, df.columns.get_loc('loss')]) / days
                if loss == 0:
                    df.iloc[c, df.columns.get_loc(rsi_str)] = 100 - (100 / (1 + gain / 0.0000000001))
                else:
                    df.iloc[c, df.columns.get_loc(rsi_str)] = 100 - (100 / (1 + gain / loss))
        else:
            # Z nejakeho magickeho dovodu to blbne ak su days<13 a len(df)> +-2600
            df[avg_gain_str] = rma(df.gain[n + 1:].to_numpy(), n, np.nansum(df.gain.to_numpy()[:n + 1]) / n)
            df[avg_loss_str] = rma(df.loss[n + 1:].to_numpy(), n, np.nansum(df.loss.to_numpy()[:n + 1]) / n)
            # df2['rs'] = df.avg_gain / df.avg_loss
            # df[rsi_str] = 100 - (100 / (1 + df.rs))
            df[rsi_str] = 100 - (100 / (1 + df[avg_gain_str] / df[avg_loss_str]))
        min_max_rsi = [df[rsi_str].min(), df[rsi_str].max()]
        print(min_max_rsi)
    # print(df[:50].to_string())

    # https://blog.quantinsti.com/build-technical-indicators-in-python/#roc
    # ROC = [(Close price today - Close price “n” day’s ago) / Close price “n” day’s ago))]
    roc_str = 'roc_' + roc
    if roc_str not in df.index:
        df[roc_str] = df[price_col_name].diff(roc) / df[price_col_name].shift(roc)


def ichimok(strat=1, conv_ln=9, base_ln=26, lag=52, displace=26):
    # https://stackoverflow.com/questions/28477222/python-pandas-calculate-ichimoku-chart-components
    # https://www.youtube.com/watch?v=WZBM-upkBm8     4:57

    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df

    conv_line_name = 'tenkan_sen_' + str(conv_ln)
    base_line_name = 'kijun_sen_' + str(base_ln)
    if conv_line_name not in df.columns:
        # Conversion line
        nine_period_high = df['High'].rolling(window=conv_ln).max()
        nine_period_low = df['Low'].rolling(window=conv_ln).min()
        df[conv_line_name] = (nine_period_high + nine_period_low) / 2

    if base_line_name not in df.columns:
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
        period26_high = df['High'].rolling(window=base_ln).max()
        period26_low = df['Low'].rolling(window=base_ln).min()
        df[base_line_name] = (period26_high + period26_low) / 2

    if strat == 4 or strat == 5:
        span_a = 'senkou_span_a_' + str(conv_ln) + '_' + str(base_ln) + '_' + str(displace)
        span_b = 'senkou_span_b_' + str(lag) + '_' + str(displace)
        # Tradingview ma o den posunutie menej
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
        df[span_a] = ((df[conv_line_name] + df[base_line_name]) / 2).shift(displace-1)

        if span_b not in df.columns:
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
            period52_high = df['High'].rolling(window=lag).max()
            period52_low = df['Low'].rolling(window=lag).min()
            df[span_b] = ((period52_high + period52_low) / 2).shift(displace-1)
            # Displace je 2x rychlejsi ako porovnavat n riadkov do predu/zadu

    lagging_span = 'chikou_span_' + str(displace)
    if strat == 2 and lagging_span not in df.columns:
        # The most current closing price plotted 22 time periods behind (optional)
        # 22 according to investopedia. Tradingview ma o den posunutie menej a aj tak nepouzivam pri rozhodovani
        # df[lagging_span+'_0'] = df[price_col_name].shift(-displace+1)
        # musim posunut naopak kvoli zjednoduseniu ratania
        df[lagging_span] = df[price_col_name].shift(displace-1)

    # i = talib.Ichimoku(df)
    # i.build(20, 60, 120, 30)
    # df.plot()
    # print(df[0:50].to_string())

    '''Strategie:
        1. conv krizuje base nad cloudom strong buy, pod cloudom weak buy   (Naj zisk 590)
        2. conv krizuje base, lagging span ako filter. Buy ak je lagging vyssie ako cena a Sell ak je nizsie 
        4. Cloud breakout - buy ak cena je prebije cloud nahor   (Naj zisk 727)
        5. ak cloudova ciara prebije druhu a cena je nad cloudom > strong buy, to iste ale cena pod cloudom weak buy
        Pouziva sa v trending markete lebo v nontrending dava vela falosnych signalov'''

    if strat == 1:
        for c, i in enumerate(df.index, 0):
            # price = df2[price_col_name][i]
            # print(df[c-1:c+1].to_string())
            if not bght and df[conv_line_name][i] > df[base_line_name][i]:
                # if df[price_col_name][i] > df['senkou_span_a'][i] and df[price_col_name][i] > df['senkou_span_b'][i]:
                entry_exit_points.append([c + 1, i, "buy"])
                rough_profit /= df[price_col_name][i]
                bght = True
            elif bght and df[conv_line_name][i] < df[base_line_name][i]:
                entry_exit_points.append([c + 1, i, "sell"])
                rough_profit *= df[price_col_name][i]
                bght = False
    elif strat == 2:
        for c, i in enumerate(df.index, 0):
            if not bght and df[conv_line_name][i] > df[base_line_name][i] and df[price_col_name][i] > df[lagging_span][i]:
                entry_exit_points.append([c + 1, i, "buy"])
                rough_profit /= df[price_col_name][i]
                bght = True
            elif bght and df[conv_line_name][i] < df[base_line_name][i] and df[price_col_name][i] < df[lagging_span][i]:
                entry_exit_points.append([c + 1, i, "sell"])
                rough_profit *= df[price_col_name][i]
                bght = False
    elif strat == 4:
        for c, i in enumerate(df.index, 0):
            if not bght and df[price_col_name][i] > df[span_a][i] and df[price_col_name][i] > df[span_b][i]:
                entry_exit_points.append([c + 1, i, "buy"])
                rough_profit /= df[price_col_name][i]
                bght = True
            elif bght and df[price_col_name][i] < df[span_a][i] and df[price_col_name][i] < df[span_b][i]:
                entry_exit_points.append([c + 1, i, "sell"])
                rough_profit *= df[price_col_name][i]
                bght = False
    elif strat == 5:
        for c, i in enumerate(df.index, 0):
            if not bght and df[span_a][i] > df[span_b][i]:
                entry_exit_points.append([c + 1, i, "buy"])
                rough_profit /= df[price_col_name][i]
                bght = True
            elif bght and df[span_a][i] < df[span_b][i]:
                entry_exit_points.append([c + 1, i, "sell"])
                rough_profit *= df[price_col_name][i]
                bght = False

    if bght:
        entry_exit_points.append([c+1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = str(strat) + "Ichimoku Conv:" + str(conv_ln) + ' Base:' + str(base_ln) + ' Senkou_posun:' + str(lag) + ' Displace:' + str(displace)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit-1, len(entry_exit_points)


def bollinger(ma=20, dwn=2, up=2):
    ''' https://www.youtube.com/watch?v=KxlUFKPCxeg
        https://www.youtube.com/watch?v=2G9fHBEsauE
    Middle band 20MA
    krajne bandy su 2 standard deviaton from middle'''

    global df
    bght = False
    entry_exit_points = []
    rough_profit = 1
    sma_str = "Sma_" + str(ma)
    stdev_str = 'Stdev_' + str(ma)
    lower_str = 'LowerBand_' + str(ma) + '_' + str(dwn)
    upper_str = 'UpperBand_' + str(ma) + '_' + str(up)
    if sma_str not in df.columns:
        df[sma_str] = df.iloc[:, price_col_indx].rolling(window=ma).mean()
    if stdev_str not in df.columns:
        # df[stdev_str] = df[sma_str].rolling(window=ma).std()
        df[stdev_str] = df[price_col_name].rolling(window=ma).std()
    if lower_str not in df.columns:
        df[lower_str] = df[sma_str] - (df[stdev_str] * dwn)
    if upper_str not in df.columns:
        df[upper_str] = df[sma_str] + (df[stdev_str] * up)

    # print(df[:50].to_string())
    # print(df[8700:].to_string())
    '''fb[[price_col_name, sma_str, upper_str, lower_str]].plot(figsize=(12, 6))
    plt.title('30 Day Bollinger Band for Facebook')
    plt.ylabel('Price (USD)')
    plt.show();'''

    # 1 Kup lower, sell upper
    # 2 Kup lower, sell middle
    for c, i in enumerate(df.index, 0):
        price = df[price_col_name][i]
        if df[sma_str][i] != df[sma_str][i]:
            continue
        if not bght and price < df[lower_str][i]:
            # print(df[c-3:c+3].to_string())
            if df[sma_str][i] != df[sma_str][i]:
                print('nan')
            entry_exit_points.append([c + 1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif bght and price > df[upper_str][i]:
            entry_exit_points.append([c + 1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False
    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "BollingerBands SMA: " + str(ma) + ' DWN: ' + str(dwn) + ' UP: ' + str(up)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit-1, len(entry_exit_points)


def renko():
    return 0


def macd(strat=1, shrt=12, lng=26, mac=9):
    # https://www.youtube.com/watch?v=kz_NJERCgm8&t=215s
    # https://tradingsim.com/blog/macd/
    # https://www.youtube.com/watch?v=9fjs8FeLMJk

    global df
    # df = df[:150]
    bght = False
    entry_exit_points = []
    rough_profit = 1
    # df[price_col_name] = round(df[price_col_name],2)

    '''plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12.2, 4.5))
    plt.plot(df['Close'], label='Close')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Price USD ($)')
    plt.show()'''

    shrt_ema_str = 'short_ema_' + str(shrt)
    lng_ema_str = 'long_ema_' + str(lng)
    macd_str = 'macd_' + str(shrt) + '_' + str(lng)
    signal_str = 'signal_' + str(shrt) + '_' + str(lng) + '_' + str(mac)
    if shrt_ema_str not in df.columns:
        df[shrt_ema_str] = df[price_col_name].ewm(span=shrt, adjust=False, min_periods=shrt).mean()
        # df[shrt_ema_str] = df.iloc[:, price_col_indx].ewm(span=shrt, adjust=False).mean()
    if lng_ema_str not in df.columns:
        df[lng_ema_str] = df[price_col_name].ewm(span=lng, adjust=False, min_periods=lng).mean()
    if macd_str not in df.columns:
        df[macd_str] = df[shrt_ema_str] - df[lng_ema_str]
    if strat != 2:
        df[signal_str] = df[macd_str].ewm(span=mac, adjust=False, min_periods=mac).mean()

    # print(df[:70].to_string())
    # print(df[-70:].to_string())

    '''plt.figure(figsize=(12.2, 4.5))
    # plt.plot(df.index, df[shrt_ema_str], label='Short', color='yellow')
    # plt.plot(df.index, df[lng_ema_str], label='Long', color='green')
    plt.plot(df.index, df[macd_str], label='MSFT MACD', color='red')
    plt.plot(df.index, signal, label='Signal Line', color='blue')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.show()'''
    if strat == 1:
        for c, i in enumerate(df.index, 0):
            # price = df[price_col_name][i]
            if df[signal_str][i] != df[signal_str][i]:
                continue
            if not bght and df[macd_str][i] > df[signal_str][i]:
                entry_exit_points.append([c + 1, i, "buy"])
                rough_profit /= df[price_col_name][i]
                bght = True
            elif bght and df[macd_str][i] < df[signal_str][i]:
                entry_exit_points.append([c + 1, i, "sell"])
                rough_profit *= df[price_col_name][i]
                bght = False
    elif strat == 2:
        for c, i in enumerate(df.index, 0):
            # price = df[price_col_name][i]
            if df[macd_str][i] != df[macd_str][i]:
                continue
            if not bght and df[macd_str][i] > 0:
                entry_exit_points.append([c + 1, i, "buy"])
                rough_profit /= df[price_col_name][i]
                bght = True
            elif bght and df[macd_str][i] < 0:
                entry_exit_points.append([c + 1, i, "sell"])
                rough_profit *= df[price_col_name][i]
                bght = False
    elif strat == 3:
        for c, i in enumerate(df.index, 0):
            # price = df[price_col_name][i]
            if df[signal_str][i] != df[signal_str][i]:
                continue
            if not bght and df[signal_str][i] > 0:
                entry_exit_points.append([c + 1, i, "buy"])
                rough_profit /= df[price_col_name][i]
                bght = True
            elif bght and df[signal_str][i] < 0:
                entry_exit_points.append([c + 1, i, "sell"])
                rough_profit *= df[price_col_name][i]
                bght = False

    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = str(strat) + "MACD Short " + str(shrt) + ' Long ' + str(lng) + ' MACD ' + str(mac)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit-1, len(entry_exit_points)


def stochastic(days=14, roll_window=3):
    # https://www.youtube.com/watch?v=AnPXC4j5buw
    # https://www.learnpythonwithrune.org/pandas-calculate-the-stochastic-oscillator-indicator-for-stocks/
    # https://www.youtube.com/watch?v=7jMbCTj17zQ&t=669s
    # KOmbinuje stochastic s 200ema, neouziva ho ako rsi a cita ho rozne podla toho ci je trh v uptrende/downtrende alebo sideways
    # Na tradingview ho este zaokruhluju
    high_str = str(days) + '_high'
    low_str = str(days) + '_low'
    if high_str not in df.columns:
        df[high_str] = df['High'].rolling(days).max()
        df[low_str] = df['Low'].rolling(days).min()
        df['%K'] = (df[price_col_name] - df[low_str]) * 100 / (df[high_str] - df[low_str])
    df['%D'] = df['%K'].rolling(roll_window).mean()
    print(df[:70].to_string())

    return 0


def my_minmax(period, proximity, s_loss):
    # V podstate Donchian Channels
    s_loss = 1 - s_loss/100
    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df

    bottom_line_str = 'bottom_' + str(period)
    top_line_str = 'top_' + str(period)
    if bottom_line_str not in df.columns:
        df[bottom_line_str] = df['Low'].rolling(window=period).min().shift(1)
        df[top_line_str] = df['High'].rolling(window=period).max().shift(1)
        # print(df[:70].to_string())

    for c, i in enumerate(df.index, 0):
        # price = df2[price_col_name][i]
        # print(df[c-1:c+1].to_string())
        if not bght and df['Low'][i] <= df[bottom_line_str][i]*(1+proximity):
            if df[top_line_str][i]*(1-proximity) / df[bottom_line_str][i]*(1+proximity) < 1.01:
                continue
            # if df[price_col_name][i] > df['senkou_span_a'][i] and df[price_col_name][i] > df['senkou_span_b'][i]:
            entry_exit_points.append([c, i, "buy", df[bottom_line_str][i]*(1+proximity)])
            rough_profit /= df[bottom_line_str][i]*(1+proximity)
            bght = True
            if s_loss == 1:
                stop_loss = 2 * df[bottom_line_str][i] - df[top_line_str][i]
            else:
                stop_loss = s_loss * df[bottom_line_str][i]
            take_profit = 100000
        elif bght and df['High'][i] >= df[top_line_str][i]*(1-proximity):
            entry_exit_points.append([c, i, "sell", stop_loss, df[top_line_str][i]*(1-proximity)])
            rough_profit *= df[top_line_str][i]*(1-proximity)
            bght = False
    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "MinMax Period: " + str(period) + ' Proximity: ' + str(proximity)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit-1, len(entry_exit_points)


def cci(ndays=20, lw=-100, hi=100):
    # Comodity Channel Index
    # Trading view ma uplne ine hodnoty :/
    # https://blog.quantinsti.com/build-technical-indicators-in-python/
    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df

    cci_str = 'cci_' + str(ndays)
    if cci_str not in df.columns:
        tp = (df['High'] + df['Low'] + df[price_col_name]) / 3
        # df['tp'] = tp
        # df['tp_sma'] = df.iloc[:, df.columns.get_loc('tp')].rolling(window=ndays).mean()
        # df["tp_sma"] = df.iloc[:, df.columns.get_loc('tp')].ewm(span=ndays, adjust=False).mean()
        # df['tp-sma'] = df['tp']-df['tp_sma']
        # df['tp_std'] = df['tp'].rolling(window=ndays).std()
        # df['my_cci'] = df['tp-sma']/(0.015 * df['tp_std'])

        # cci1 = pd.Series((tp - tp.rolling(ndays).mean()) / (0.015 * tp.rolling(ndays).std()), name=cci_str)
        # cci1 = pd.Series((tp - pd.rolling_mean(tp, ndays)) / (0.015 * pd.rolling_std(tp, ndays)), name=cci_str)
        # cci1 = pd.Series((tp - pd.rolling_mean(tp, ndays)) / (0.015 * pd.rolling_std(tp, ndays)), name=cci_str)
        df[cci_str] = pd.Series((tp - tp.rolling(ndays).mean()) / (0.015 * tp.rolling(ndays).std()), name=cci_str)
        # df = df.join(cci1)
        # print(df[8040:8050].to_string())

    for c, i in enumerate(df.index, 0):
        if not bght and df[cci_str][i] < lw:
            entry_exit_points.append([c + 1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif bght and df[cci_str][i] > hi:
            entry_exit_points.append([c + 1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False

    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "CCI days " + str(ndays) + ' Low ' + str(lw) + ' High ' + str(hi)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit-1, len(entry_exit_points)


def eom(days=14):
    # EOM - Ease of Movement
    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df
    eom_str = 'EOM_' + str(days)
    if eom_str not in df.columns:
        dm = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
        br = (df['Volume'] / 100000000) / (df['High'] - df['Low'])
        evm = dm / br
        df[eom_str] = pd.Series(evm.rolling(days).mean(), name=eom_str)
        # print(df[8040:8050].to_string())
        # print(df[8000:8050].to_string())

    for c, i in enumerate(df.index, 0):
        if not bght and df[eom_str][i] > 0:
            entry_exit_points.append([c + 1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif bght and df[eom_str][i] < 0:
            entry_exit_points.append([c + 1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False

    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "EOM days " + str(days)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit - 1, len(entry_exit_points)


def mfi(days=14, lw=20, hi=80):
    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df
    global min_max_rsi
    mfi_str = 'MFI_' + str(days)
    if mfi_str not in df.columns:
        tp = (df['High'] + df['Low'] + df[price_col_name]) / 3
        money_flow = tp * df['Volume']
        # RSI
        # df['change'] = df[price_col_name].diff()
        # df['gain'] = df.change.mask(df.change < 0, 0.0)
        # df['loss'] = -df.change.mask(df.change > 0, -0.0)

        # Get all of the positive and negative money flow
        positive_flow = [np.nan]
        negative_flow = [np.nan]
        # Loop through typical price calculations
        for i in range(1, len(tp)):
            if tp[i] > tp[i - 1]:
                positive_flow.append(money_flow[i])  # i-1
                negative_flow.append(0)
            elif tp[i] < tp[i - 1]:
                negative_flow.append(money_flow[i])  # i-1
                positive_flow.append(0)
            else:
                positive_flow.append(0)
                negative_flow.append(0)

        positive_mf = (days-1) * [np.nan]
        negative_mf = (days-1) * [np.nan]
        for i in range(days-1, len(positive_flow)):
            positive_mf.append(sum(positive_flow[i + 1 - days:i + 1]))
        for i in range(days-1, len(negative_flow)):
            negative_mf.append(sum(negative_flow[i + 1 - days:i + 1]))

        df[mfi_str] = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))
        # print(df[8000:8050].to_string())
        min_max_rsi = [df[mfi_str].min(), df[mfi_str].max()]
        print(min_max_rsi)

    for c, i in enumerate(df.index, 0):
        # price = df2[price_col_name][i]
        if not bght and lw > df[mfi_str][i]:
            entry_exit_points.append([c + 1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif bght and hi < df[mfi_str][i]:
            entry_exit_points.append([c + 1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False

    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "MFI Days " + str(days) + ' Buy_S ' + str(lw) + ' Sell_S ' + str(hi)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit - 1, len(entry_exit_points)


def vortex(days=14):
    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df
    global min_max_rsi
    vrx_str_p = 'vi+_' + str(days)
    vrx_str_m = 'vi-_' + str(days)
    if vrx_str_p not in df.columns:
        y_hi = df['High'].shift(1)
        y_lo = df['Low'].shift(1)
        y_cls = df[price_col_name].shift(1)
        vm_pls = df['High'] - y_lo
        vm_mns = df['Low'] - y_hi
        vm_pls_sum = vm_pls.rolling(min_periods=days, window=days).sum()
        vm_mns_sum = vm_mns.rolling(min_periods=days, window=days).sum()
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = (df['Low'] - y_cls).abs()
        df['tr3'] = (df['High'] - y_cls).abs()
        # tr = max(df['tr1'], df['tr2'], df['tr3'])
        tr = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        tr_sum = tr.rolling(min_periods=days, window=days).sum()
        # tr = pd.Series.iloc[:, [tr1, tr2, tr3]].max(axis=1, skipna=True)
        df[vrx_str_p] = vm_pls_sum / tr_sum
        df[vrx_str_m] = vm_mns_sum.abs() / tr_sum
        # print(df[:50].to_string())

    for c, i in enumerate(df.index, 0):
        if not bght and df[vrx_str_p][i] > df[vrx_str_m][i]:
            entry_exit_points.append([c + 1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif bght and df[vrx_str_p][i] < df[vrx_str_m][i]:
            entry_exit_points.append([c + 1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False

    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "Vortex Days " + str(days)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit - 1, len(entry_exit_points)


def alligator(strat=1, lps=5, tth=8, jw=13):
    # def alligator(strat=1, first_shift=3, lps=5, tth=8, jw=13):
    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df
    lips_str = 'lips_' + str(lps)  # + "_" + str(first_shift)
    teeth_str = 'teeth_' + str(tth)  # + "_" + str(lps)
    jaw_str = 'jaw_' + str(jw)  # + "_" + str(tth)
    if 'md' not in df.columns:
        median_prc = (df['High'] + df['Low']) / 2
        # median_prc_sum = median_prc.rolling(min_periods=lps, window=lps).sum()
        df['md'] = median_prc
        # df[lips_str] = df.iloc[:, median_prc].rolling(window=lps).mean()
        # df[lips_str+"x"] = median_prc.rolling(window=lps).mean()
    if lips_str not in df.columns:
        bar = []
        for c, i in enumerate(df.index, 0):
            if c < lps - 1:
                bar.append(np.nan)
            elif c == lps - 1:
                foo = 0
                for d in range(lps):
                    foo += df.iloc[c - d, df.columns.get_loc('md')]
                foo /= lps
                bar.append(foo)
            else:
                foo = (foo * (lps - 1) + df['md'][c]) / lps
                bar.append(foo)
        df[lips_str] = bar
        df[lips_str] = df[lips_str].shift(3)  # first_shift
    if teeth_str not in df.columns:
        bar = []
        for c, i in enumerate(df.index, 0):
            if c < tth - 1:
                bar.append(np.nan)
            elif c == tth - 1:
                foo = 0
                for d in range(tth):
                    foo += df.iloc[c - d, df.columns.get_loc('md')]
                foo /= tth
                bar.append(foo)
            else:
                foo = (foo * (tth - 1) + df['md'][c]) / tth
                bar.append(foo)
        df[teeth_str] = bar
        df[teeth_str] = df[teeth_str].shift(5)  # lps
    if jaw_str not in df.columns:
        bar = []
        for c, i in enumerate(df.index, 0):
            if c < jw - 1:
                bar.append(np.nan)
            elif c == jw - 1:
                foo = 0
                for d in range(jw):
                    foo += df.iloc[c - d, df.columns.get_loc('md')]
                foo /= jw
                bar.append(foo)
            else:
                foo = (foo * (jw - 1) + df['md'][c]) / jw
                bar.append(foo)
        df[jaw_str] = bar
        df[jaw_str] = df[jaw_str].shift(8)  # tth
    # print(df[:50].to_string())

    if strat == 1:
        for c, i in enumerate(df.index, 0):
            if not bght and df[lips_str][i] > df[teeth_str][i] and df[lips_str][i] > df[jaw_str][i]:
                entry_exit_points.append([c + 1, i, "buy"])
                rough_profit /= df[price_col_name][i]
                bght = True
            elif bght and df[lips_str][i] < df[teeth_str][i] and df[lips_str][i] < df[jaw_str][i]:
                entry_exit_points.append([c + 1, i, "sell"])
                rough_profit *= df[price_col_name][i]
                bght = False
    elif strat == 2:
        for c, i in enumerate(df.index, 0):
            if not bght and df[lips_str][i] > df[teeth_str][i] and \
                    df[jaw_str][i] < df[lips_str][i] < df[price_col_name][i]:
                entry_exit_points.append([c + 1, i, "buy"])
                rough_profit /= df[price_col_name][i]
                bght = True
            elif bght and df[lips_str][i] < df[teeth_str][i] and \
                    df[jaw_str][i] > df[lips_str][i] > df[price_col_name][i]:
                entry_exit_points.append([c + 1, i, "sell"])
                rough_profit *= df[price_col_name][i]
                bght = False

    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    # strategy_variables = str(strat) + "Alligator first " + str(first_shift) + ' Lips ' + str(lps) + ' Teeth ' + str(tth) + ' Jaw ' + str(jw)
    strategy_variables = str(strat) + "Alligator Lips " + str(lps) + ' Teeth ' + str(tth) + ' Jaw ' + str(jw)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit - 1, len(entry_exit_points)


def chaikin(shrt=3, lng=10):
    # Ine cisla ako na tradingview
    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df
    co_str = 'co_' + str(shrt) + "_" + str(lng)
    adl_shrt = 'adl_' + str(shrt)
    adl_lng = 'adl_' + str(lng)
    if 'm_line' not in df.columns:
        m_line = (2*df[price_col_name] - df['Low'] - df['High']) * df['Volume'] / (df['High'] - df['Low'])
        # df['m_line'] = m_line
        df['adl_line'] = m_line + m_line.shift(1)

        '''bar = []
        for c, i in enumerate(df.index, 0):
            if c == 0:
                bar.append(m_line[0])
            else:
                bar.append(bar[-1] + m_line[c])
        df['adl_line2'] = bar'''

    if adl_shrt not in df.columns:
        df[adl_shrt] = df['adl_line'].ewm(span=shrt, adjust=False, min_periods=shrt).mean()
    if adl_lng not in df.columns:
        df[adl_lng] = df['adl_line'].ewm(span=lng, adjust=False, min_periods=lng).mean()
    df[co_str] = df[adl_shrt] - df[adl_lng]
    # print(df[:150].to_string())

    for c, i in enumerate(df.index, 0):
        if not bght and df[co_str][i] > 0:
            entry_exit_points.append([c + 1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif bght and df[co_str][i] < 0:
            entry_exit_points.append([c + 1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False

    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "Chaikin Short " + str(shrt) + " Long " + str(lng)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit - 1, len(entry_exit_points)


def parabolic_sar(iaf=0.02, maxaf=0.2):
    # Ine cisla ako na tradingview
    if maxaf < iaf:
        return 0, 100
    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df
    barsdata = df
    length = len(barsdata)
    high = list(barsdata['High'])
    low = list(barsdata['Low'])
    close = list(barsdata['Close'])
    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    # ep = low[0]
    hp = high[0]
    lp = low[0]
    psar_str = 'psar_' + str(iaf) + '_' + str(maxaf)
    for i in range(1, length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    df[psar_str] = psar
    # df['psarbear'] = psarbear
    # df['psarbull'] = psarbull
    # print(df[:30].to_string())

    for c, i in enumerate(df.index, 0):
        if not bght and df[psar_str][i] < df[price_col_name][i]:
            entry_exit_points.append([c + 1, i, "buy"])
            rough_profit /= df[price_col_name][i]
            bght = True
        elif bght and df[psar_str][i] > df[price_col_name][i]:
            entry_exit_points.append([c + 1, i, "sell"])
            rough_profit *= df[price_col_name][i]
            bght = False

    if bght:
        entry_exit_points.append([c + 1, i, "sell"])
        rough_profit *= df[price_col_name][i]
    strategy_variables = "ParabolicSAR iAF " + str(iaf) + ' maxAF ' + str(maxaf)
    strategy_end(entry_exit_points, rough_profit - 1, strategy_variables)
    return rough_profit - 1, len(entry_exit_points)


def parabolic_sar1(iaf=0.02, maxaf=0.2):
    entry_exit_points = []
    rough_profit = 1
    bght = False
    global df
    iaf = 1
    bull = True
    last_high = 0
    last_low = 0
    psar = []
    af = iaf
    for c, i in enumerate(df.index, 0):
        if df['Low'][i] > last_low:
            last_low = df['Low'][i]
        if df['High'][i] > last_high:
            last_high = df['High'][i]
        if c == 0:
            psar.append(last_low)
            last_psar = last_low
            # af += iaf
            continue
        if bull:
            last_psar = last_psar + af * (df['High'][c-1] - last_psar)
            psar.append(last_psar)
            # af += iaf
        else:
            continue

    df['psar'] = psar
    print(df[:30].to_string())
    return 0, 0


def run_test(df1):
    global df
    global price_col_name
    global price_col_indx
    global sell_on_exit_points
    global short_report
    global min_max_rsi
    global strategy
    global output_above_profit
    global output_above_trades

    output_above_profit = 100  # 250
    output_above_trades = 70
    # sell_on_exit_points = False
    short_report = True
    df = MSM.load_pickle(df1)
    price_col_name = 'Close'
    price_col_indx = df.columns.get_loc(price_col_name)
    incr = 5
    # merged.at['2021-01-15', 'Volume'] = 999
    # merged.iloc[2,3] = 32
    # merged.loc['2021-02-08','Volume'] = 2

    strat_names = {1: 'rbw',
                   2: 'sma',
                   3: 'rsi',
                   4: 'bollinger',
                   51: 'ichimoku1',
                   52: 'ichimoku2',
                   54: 'ichimoku4',
                   55: 'ichimoku5',
                   61: 'macd1',
                   62: 'macd2',
                   63: 'macd3',
                   7: 'minmax',
                   8: 'stochastic',
                   9: 'cci',
                   10: 'eom',
                   11: 'mfi',
                   12: 'vortex',
                   131: 'alligator1',
                   132: 'alligator2',
                   14: 'chaikin',
                   15: 'parabolic_sar',
                   }
    strat_vars = {'rbw': [0, 4, 1, 47, 2, 147, 3, 200, 50, 200],
                  'sma': [0, 1, 350, 600, 0, 1, 0, 1, 0, 1],
                  'rsi': [0, 3, 4, 158, 0, 100, 0, 101, 0, 1],
                  'bollinger': [0, 3, 4, 21, 0, 20, 15, 30, 0, 1],
                  'ichimoku1': [1, 2, 9, 18, 10, 34, 0, 1, 0, 1],
                  'ichimoku2': [2, 3, 9, 15, 10, 31, 0, 1, 125, 210],
                  'ichimoku4': [4, 4, 9, 18, 10, 33, 1, 26, 178, 200],
                  'ichimoku5': [5, 4, 9, 18, 10, 34, 1, 53, 26, 190],  # 190
                  'macd1': [1, 3, 1, 100, 1, 101, 2, 100, 0, 1],
                  'macd2': [2, 2, 1, 65, 1, 250, 2, 3, 0, 1],
                  'macd3': [3, 3, 1, 30, 1, 50, 2, 200, 0, 1],
                  'minmax': [0, 3, 3, 300, 0, 200, 0, 10, 0, 1],
                  'stochastic': [0, 2, 0, 300, 0, 200, 0, 1, 0, 1],
                  'cci': [0, 3, 1, 180, -100, 200, 80, 200, 0, 1],
                  'eom': [0, 1, 1, 600, 0, 1, 0, 1, 0, 1],
                  'mfi': [0, 3, 2, 158, 0, 100, 0, 101, 0, 1],
                  'vortex': [0, 1, 1, 600, 0, 1, 0, 1, 0, 1],
                  # 'alligator1': [1, 4, 1, 200, 2, 200, 3, 80, 4, 200],
                  # 'alligator1': [1, 4, 1, 70, 2, 60, 3, 80, 4, 200],
                  # 'alligator2': [2, 4, 1, 70, 2, 60, 3, 80, 4, 200],
                  'alligator1': [1, 3, 1, 100, 2, 150, 3, 100, 0, 1],
                  'alligator2': [2, 3, 1, 100, 2, 150, 3, 100, 0, 1],
                  'chaikin': [0, 2, 1, 600, 2, 601, 0, 1, 0, 1],
                  'parabolic_sar': [0, 2, 45, 1000, 0, 20, 0, 1, 0, 1],
                  }
    strategy = strat_names[15]  # <===== Tu vyberas strategiu
    v = strat_vars[strategy]
    loops = v[1]
    for i1 in range(v[2], v[3], incr):
        if which_iter:
            if i1 != which_iter[0]:
                continue
        min_max_rsi = [0, 100]  # rsi
        skip = 0
        for i2 in range(v[4], v[5], incr):  # v[4], v[5], incr
            print(i1, i2)
            if 'rsi' == strategy or 'mfi' == strategy:
                if i2 < min_max_rsi[0]:
                    continue
                elif i2 > min_max_rsi[1]:
                    break
            if i2 <= i1:
                i2_lst = ['macd', 'rbw', 'alligator', 'chaikin']
                if strategy in i2_lst:
                    continue
            if loops > 2:
                skip = 0
                df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']]
            for i3 in range(v[6], v[7], incr):
                # print(i1, i2, i3)
                if which_iter:
                    if i2 < which_iter[1] or (i2 == which_iter[1] and i3 < which_iter[2]):
                        continue
                if 'rsi' == strategy or 'mfi' == strategy:
                    if i3 < min_max_rsi[0] or i3 < i2:
                        continue
                    elif i3 > min_max_rsi[1] or i2 < min_max_rsi[0]:
                        break
                if i3 <= i2:
                    i3_lst = ['rbw', 'cci', 'alligator']
                    if strategy in i3_lst:
                        continue
                if loops == 4:
                    skip = 0
                if skip and loops == 3:
                    skip -= 1
                    continue
                for i4 in range(v[8], v[9], incr):
                    if skip and loops == 4:
                        skip -= 1
                        continue
                    if i4 <= i3:
                        if 'rbw' == strategy:
                            continue
                    if 'rbw' == strategy:
                        approx_profit, nmr_trds = red_white_blue_strategy(i1, i2, i3, i4)
                    elif 'sma' == strategy:
                        approx_profit, nmr_trds = simple_moving_average(i1)
                    elif 'rsi' == strategy:
                        approx_profit, nmr_trds = rsi(i1, i2, i3)
                    elif 'ichimoku' in strategy:
                        approx_profit, nmr_trds = ichimok(v[0], i1, i2, i3, i4)
                    elif 'bollinger' == strategy:
                        approx_profit, nmr_trds = bollinger(i1, i2 / 10, i3 / 10)
                    elif 'renko' == strategy:
                        approx_profit = renko()
                    elif 'macd' in strategy:
                        approx_profit, nmr_trds = macd(v[1], i1, i2, i3)
                    elif 'minmax' == strategy:
                        approx_profit, nmr_trds = my_minmax(i1, i2/100, i3)
                    elif 'stochastic' == strategy:
                        approx_profit, nmr_trds = stochastic()
                    elif 'cci' == strategy:
                        approx_profit, nmr_trds = cci(i1, i2, i3)
                    elif 'eom' == strategy:
                        approx_profit, nmr_trds = eom(i1)
                    elif 'mfi' == strategy:
                        approx_profit, nmr_trds = mfi(i1, i2, i3)
                    elif 'vortex' == strategy:
                        approx_profit, nmr_trds = vortex(i1)
                    elif 'alligator' in strategy:
                        # approx_profit, nmr_trds = alligator(v[0], i1, i2, i3, i4)
                        approx_profit, nmr_trds = alligator(v[0], i1, i2, i3)
                    elif 'chaikin' == strategy:
                        approx_profit, nmr_trds = chaikin(i1, i2)
                    elif 'parabolic_sar' == strategy:
                        approx_profit, nmr_trds = parabolic_sar(i1 / 1000, i2 / 100)
                    if nmr_trds < output_above_trades - 10 or approx_profit < output_above_profit * 0.6:
                        skip = 0

    # df = df.iloc[ma-1:]
    # for i in df.index:
    #     print(df.iloc[:,4][i])
    #     print(df[sma_str][i])
    #     print(df.iloc[:,0][i])


tickerSymbol = 'MSFT'
option = 0
if option == 0:
    # get data on this ticker
    start = datetime.datetime(2021, 1, 13)
    start = '2021-02-12'
    end = datetime.datetime(2021, 1, 21)
    end = '2021-04-10'
    # end = datetime.datetime.now()
    # 1d, 5d, 1wk, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max  z akeho obdobia ma zbierat data
    period = ''
    # interval:  1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    # 1m len poslednych 7 dni, 2m-1h len poslednych 60 dni
    intrvl = '2m'  # '2m'  '1d
    get_data(tickerSymbol, start, end, period, intrvl)
elif option == 1:
    # merges dataframes
    df = merge_dfs('MSFT 161.pkl', 'MSFT 1613514215.pkl')
    MSM.save_pickle(tickerSymbol + '.pkl', df)
elif option == 2:
    # pickkle to txt
    fl = 'MSFT.pkl'
    df1 = MSM.load_pickle(fl)
    fl = fl.replace('.pkl', '.txt')
    with open(fl, "w") as fo:
        fo.write(df1.to_string())
elif option == 3:
    # runs test, but go to def and choose which strategy. 21 Strategies and billions of settings
    # run_test('MSFT 1D 1613514215.pkl')
    run_test('MSFT 1d max 1613686025.pkl')
elif option == 4:
    # it can run multiple tests in paralell
    # run_test('MSFT 1D 1613514215.pkl')
    # for t in range(9, 18):
    for t in range(1, 4):
        # _thread.start_new_thread(run_test_t, ('MSFT 1d max 1613686025.pkl', t, ))
        _thread.start_new_thread(run_test_t, ('MSFT 1d max 1613686025.pkl', t,))
        # threading.Thread(target=run_test_t, args=('MSFT 1d max 1613686025.pkl', t, )).start()

elif option == 5:
    # show graph
    # import finplot as fplt
    # fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
    # fplt.show()
    pass
    print(21, '//', 3, 21//3)
    print(21, '%', 3, 21 % 3)
foo = input('Press Enter to finish...\n')
