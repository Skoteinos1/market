# Analysis of Sotock Market Indicators.

![Stock Market](https://github.com/Skoteinos1/market/blob/main/stockmarket.jpg)

Ever wondered which indicator is the best and at what settings? This code will answer that. There are billions of combinations... It will run one week for every ticker. Anyway short answer is RSI, not at default settings, do fundamental before technical and your money is at risk.

Settings:<br>
option = 0 - Downloads data on ticker<br>
option = 1 - Merges dataframes<br>
option = 2 - Pickle to txt<br>
option = 3 - Runs test, but go to def and choose which strategy. 21 Strategies and billions of settings<br>
    Tested strategies:<br>
    strat_names = {1: 'rbw',<br>
                   2: 'sma',<br>
                   3: 'rsi',<br>
                   4: 'bollinger',<br>
                   51: 'ichimoku1',<br>
                   52: 'ichimoku2',<br>
                   54: 'ichimoku4',<br>
                   55: 'ichimoku5',<br>
                   61: 'macd1',<br>
                   62: 'macd2',<br>
                   63: 'macd3',<br>
                   7: 'minmax',<br>
                   8: 'stochastic',<br>
                   9: 'cci',<br>
                   10: 'eom',<br>
                   11: 'mfi',<br>
                   12: 'vortex',<br>
                   131: 'alligator1',<br>
                   132: 'alligator2',<br>
                   14: 'chaikin',<br>
                   15: 'parabolic_sar',<br>
                   }<br>
    strat_vars = {'rbw': [0, 4, 1, 47, 2, 147, 3, 200, 50, 200],<br>
                  'sma': [0, 1, 350, 600, 0, 1, 0, 1, 0, 1],<br>
                  'rsi': [0, 3, 4, 158, 0, 100, 0, 101, 0, 1],<br>
                  'bollinger': [0, 3, 4, 21, 0, 20, 15, 30, 0, 1],<br>
                  'ichimoku1': [1, 2, 9, 18, 10, 34, 0, 1, 0, 1],<br>
                  'ichimoku2': [2, 3, 9, 15, 10, 31, 0, 1, 125, 210],<br>
                  'ichimoku4': [4, 4, 9, 18, 10, 33, 1, 26, 178, 200],<br>
                  'ichimoku5': [5, 4, 9, 18, 10, 34, 1, 53, 26, 190],  # 190<br>
                  'macd1': [1, 3, 1, 100, 1, 101, 2, 100, 0, 1],<br>
                  'macd2': [2, 2, 1, 65, 1, 250, 2, 3, 0, 1],<br>
                  'macd3': [3, 3, 1, 30, 1, 50, 2, 200, 0, 1],<br>
                  'minmax': [0, 3, 3, 300, 0, 200, 0, 10, 0, 1],<br>
                  'stochastic': [0, 2, 0, 300, 0, 200, 0, 1, 0, 1],<br>
                  'cci': [0, 3, 1, 180, -100, 200, 80, 200, 0, 1],<br>
                  'eom': [0, 1, 1, 600, 0, 1, 0, 1, 0, 1],<br>
                  'mfi': [0, 3, 2, 158, 0, 100, 0, 101, 0, 1],<br>
                  'vortex': [0, 1, 1, 600, 0, 1, 0, 1, 0, 1],<br>
                  # 'alligator1': [1, 4, 1, 200, 2, 200, 3, 80, 4, 200],<br>
                  # 'alligator1': [1, 4, 1, 70, 2, 60, 3, 80, 4, 200],<br>
                  # 'alligator2': [2, 4, 1, 70, 2, 60, 3, 80, 4, 200],<br>
                  'alligator1': [1, 3, 1, 100, 2, 150, 3, 100, 0, 1],<br>
                  'alligator2': [2, 3, 1, 100, 2, 150, 3, 100, 0, 1],<br>
                  'chaikin': [0, 2, 1, 600, 2, 601, 0, 1, 0, 1],<br>
                  'parabolic_sar': [0, 2, 45, 1000, 0, 20, 0, 1, 0, 1],<br>
                  }<br>
    You don't have to worry about variables, software tries to go all of them, those numbers you see above are limits, otherwise it would take ages to go to everything. It takes roughly a week to test all strategies.

option = 4 - Runs multiple tests in parallel<br>
option = 5 - Show graph<br>

Warning:
You will see some ridiculous numbers, but do not go crazy and do not go all in on MSFT or anything else. Things you will learn from these tests:
1. Strategy with 10% success rate can be more profitable than strategy with 90% success rate.
2. Higher number of trades doesn't mean higher profit
3. Every moron can come up with profitable strategy. But if strategy can't beat just holding, it is worthless.
4. Finding great strategy is like finding math function which describes graph well. There is no guarantee that price action will follow your function.
5. Before trading on stock market, do fundamental analysis first and pick your best companies. Than you can start to do technical. And remember: Time in the market beats timing the market.

# Your money is at risk. Don't blame me if you lose all of them.
