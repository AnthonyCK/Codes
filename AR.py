# create and evaluate an updated autoregressive model
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
series = pd.read_excel('./michigan-history.xlsx', sheet_name=1, header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
pd.plotting.lag_plot(series)
pyplot.show()
pd.plotting.autocorrelation_plot(series)
pyplot.show()
plot_acf(series, lags = 30)
pyplot.show()

pyplot.show()
# split dataset
X = series.values
train, test = X[1:len(X)-30], X[len(X)-30:]
# train autoregression
window = 30
model = AutoReg(train, lags=30)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
