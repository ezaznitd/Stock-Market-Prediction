from flask import Flask
import time
import os
from flask_apscheduler import APScheduler
from yahoo_finance_api import YahooFinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

app = Flask(__name__)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

from views import *

#Model training part
def train_model(data_file, model_file, window):
	df = pd.read_csv('./static/' + data_file)
	n = len(df)
	training_size = int(0.9 * n)
	training_set = df.iloc[:training_size, 4:5].values

	sc = MinMaxScaler(feature_range=(0,1))
	training_set_scaled = sc.fit_transform(training_set)

	X_train = []
	y_train = []
	for i in range(window, training_size):
		X_train.append(training_set_scaled[i - window : i, 0])
		y_train.append(training_set_scaled[i, 0])
	X_train, y_train = np.array(X_train), np.array(y_train)
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
	
	model = Sequential()
	model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
	model.add(Dropout(0.2))
	model.add(LSTM(units=50,return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(units=50,return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(units=50))
	model.add(Dropout(0.2))
	model.add(Dense(units=1))
	model.compile(optimizer='adam',loss='mean_squared_error')
	model.fit(X_train,y_train,epochs=100,batch_size=32)

	model.save('./trained_model/' + model_file)

#Background task. This method will run parallally with the website and updates our traind
#for better accuracy. This background process will decrease the Machine Learning model
#training time so that our website will run faster.
def scheduled_task(task_id):
	con = sql.connect("company_list.db")
	con.row_factory = sql.Row
	curr = con.cursor()
	curr.execute("select symbol from NSE_list")
	rows = curr.fetchall()
	for i in range(len(rows)):
		ticker_sym = rows[i][0] + ".NS"
		ticker = rows[i][0]
		interval = "one-day"
		window = 60
		n = 7
		data_file = "{ticker}-{interval}.csv".format(ticker = ticker, interval = "one-day")
		model_file = "{ticker}-{interval}.h5".format(ticker = ticker, interval = "one-day")
		data = yf(ticker_sym, result_range='2000d', interval='1d').result
		df = pd.DataFrame(data)
		df.to_csv('./static/' + data_file)
		train_model(data_file, model_file, window)

if __name__ == '__main__':
	app.apscheduler.add_job(func=scheduled_task, trigger='interval', weeks=1, args=[100], id="background_process")
	app.run()
