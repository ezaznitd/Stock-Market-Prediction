from run import app
from flask import Markup, render_template, redirect, request, url_for
import pandas as pd
import os
import sqlite3 as sql
from yahoo_finance_api import YahooFinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

#Sentiment Analysis
import tweepy
import constants as ct
import preprocessor as p
import re
from textblob import TextBlob
from Tweet import Tweet

#This process will return polarity of tweets towards any company.
def get_polarity(tweets):
	tweet_list = []
	global_polarity = 0
	tw_list=[]
	pos = 0
	neg = 1
	for tweet in tweets:
		count = 20
		tw2 = tweet.full_text
		tw = tweet.full_text
		tw=p.clean(tw)
		tw=re.sub('&amp;','&',tw)
		tw=re.sub(':','',tw)
		tw=tw.encode('ascii', 'ignore').decode('ascii')
		blob = TextBlob(tw)
		polarity = 0
		for sentence in blob.sentences:
			polarity += sentence.sentiment.polarity
			if polarity>0:
				pos=pos+1
			if polarity<0:
				neg=neg+1
			global_polarity += sentence.sentiment.polarity
		if count > 0:
			tw_list.append(tw2)
		tweet_list.append(Tweet(tw, polarity))
		count=count-1
	return [tweet_list, global_polarity, pos, neg]

#sentiment analysis part using tweeter api for future stock value prediction.
def sentiment_analysis(symbol):
	auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
	auth.set_access_token(ct.access_token, ct.access_token_secret)
	user = tweepy.API(auth)
	tweets = tweepy.Cursor(user.search, q=str(symbol), tweet_mode='extended', lang='en',exclude_replies=True).items(ct.num_of_tweets)
	tweet_list, global_polarity, pos, neg = get_polarity(tweets)
	if len(tweet_list) > 0:
		global_polarity = global_polarity / len(tweet_list)
	neutral=ct.num_of_tweets-pos-neg
	if neutral<0:
		neg=neg+neutral
		neutral=20
	return [global_polarity, pos, neg, neutral]

class company:
	company_name = "Reliance Industries Limited"
	symbol = "RELIANCE.NS"

#Divide the our website manipulation and background process into two parallal process.
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/aboutus')
def about():
	return render_template('aboutUs.html')

@app.route('/contactus')
def contact():
	return render_template('contactUs.html')


#search engine
@app.route('/search', methods=['GET', 'POST'])
def search():
		name = request.form['sea']
		con = sql.connect("company_list.db")
		con.row_factory = sql.Row
		curr = con.cursor()
		curr.execute("select company_name from NSE_list where company_name like ?",("%"+name+'%',))
		rows = curr.fetchall()
		return render_template('layout.html', rows=rows)

#load company name by alphabetically order
@app.route('/load_company_database<character>')
def load_company_database(character):
	con = sql.connect("company_list.db")
	con.row_factory = sql.Row
	curr = con.cursor()
	curr.execute("select company_name from NSE_list where company_name like ?",(character+'%',))
	rows = curr.fetchall()
	return render_template('layout.html', rows=rows)

#This methods will return all the comapanies whose name group by alphabet.
@app.route('/alphabet/<character>')
def alphabet(character):
	return redirect(url_for('load_company_database', character=character))

#This method will return all the comapnies whose name start from anything except alphabet.
@app.route('/Other')
def Other():
	con = sql.connect("company_list.db")
	con.row_factory = sql.Row
	curr = con.cursor()
	curr.execute("select company_name from Other;")
	rows = curr.fetchall()
	return render_template('layout.html', rows=rows)

#Get maximum and minimum closing price.
def get_data(label, column):
	labels = []
	close_price = []
	max_value = 0.0
	min_value = 200000.0
	for i in range(1,len(label)):
		labels.append(label[i])
		close_price.append(float(column[i]))
		max_value = max(max_value, float(column[i]))
		min_value = min(min_value, float(column[i]))
	return [max_value, min_value, labels, close_price]

result_range_for_each_interval = {'1m': '1d', '2m': '1d', '5m': '5d', '15m': '5d', '30m': '1mo', '60m': '3mo', '90m': '1mo', '1d': '1y', '5d': '2y', '1wk': '5y', '1mo': '10y', '3mo': 'max'}
dataset_for_each_interval = {'1m': "_one_minute", '2m': "_two_minute", '5m': "_five_minute", '15m': "_fifteen_minute", '30m': "_thirty_minute", '60m': "_sixty_minute", '90m': "_ninety_minute", '1d': "_one_day", '5d': "_five_day", '1wk': "_one_week", '1mo': "_one_month", '3mo': "_three_month"}

#This method will return live stock data for any interval
@app.route('/interval/<name>')
def interval(name):
	data = yf(company.symbol, result_range=result_range_for_each_interval[name], interval=name, dropna='True').result
	df = pd.DataFrame(data)
	filename = company.company_name + dataset_for_each_interval[name] + ".csv"
	df.to_csv('./stock_dataset/' + filename)
	dataset = pd.read_csv('./stock_dataset/' + filename, sep=',', header=None)
	label = dataset.values[:,0]
	column = dataset.values[:,4]
	max_value, min_value, labels, close_price = get_data(label, column)
	title = company.company_name + " Stock Price in INR"
	return render_template('line_chart.html', title=title, max=max_value, min=min_value, labels=labels, values=close_price)

def reshape_list(sc, df, window):
	inputs = sc.fit_transform(df.iloc[-window:, 3:4].values)
	inputs = inputs.reshape(-1, 1)
	X = [inputs]
	X = np.array(X)
	X = np.reshape(X, (X.shape[0], X.shape[1], 1))
	return X

# gets last window size number of elements
def get_last_window_values(data, window, sc):
	df = pd.DataFrame(data)
	X = reshape_list(sc, df, window)
	Y = reshape_list(sc, df, window - 1)
	return [X, Y]

#This method will return future 7 days stock price using dataset upto today.
def predict_price(X, model):
	score = model.predict(X)
	return score

#predict stock for today and compare it with the actual stock for today
#so, `error_of_predicttion` = `predicted_stock_for_today` - `actual_stock_for_today`
#so at the time of predicting stock for tomorrow we have substruct `error_of_prediction` from predicted
#stock for tomorrow to get more closer predicted stock value to the actual stock
def get_error_for_today(sc, window, Y, model, global_polarity, close_price):
	predicted_scores_for_today = []
	Y = Y[-(window - 1):]
	temp = predict_price(Y, model)
	Y = np.append(Y, temp)
	Y = [Y]
	Y = np.array(Y)
	Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))
	predicted_scores_for_today.append(temp[-1])
	predicted_price_for_today = sc.inverse_transform(predicted_scores_for_today)
	error = predicted_price_for_today - close_price[-1] + predicted_price_for_today * global_polarity
	return error

#prediction part
def predict_future_stock(labels, close_price, ticker, label, tata_power):
	model_file = "{ticker}-{interval}.h5".format(ticker = ticker, interval = "one-day")
	window = 60
	n = 7
	d = label[len(label)-1]
	curr_date = ""
	ext = ""
	for i in range(len(d)):
		if i < 10:
			curr_date += d[i]
		else:
			ext += d[i]

	if os.path.exists('./trained_model/' + model_file):
		sc = MinMaxScaler(feature_range=(0,1))
		model = load_model('./trained_model/' + model_file)
		X, Y = get_last_window_values(tata_power, window, sc)
		inputs = X
		predicted_scores = []
		global_polarity, pos, neg, neutral = sentiment_analysis(company.symbol)
		error = get_error_for_today(sc, window, Y, model, global_polarity, close_price)
		for i in range(n):
			inputs = inputs[-window:]
			price = predict_price(inputs, model)
			inputs = np.append(inputs, price)
			inputs = [inputs]
			inputs = np.array(inputs)
			inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
			predicted_scores.append(price[-1])
			predicted_price = sc.inverse_transform(predicted_scores)
		for i in range(n):
			curr_date = (datetime.strptime(curr_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
			labels.append(curr_date + ext)
			price = predicted_price[i] + predicted_price[i] * global_polarity - error
			close_price.append(price)
	return [labels, close_price]

@app.route('/company/interval/<name>', methods=['GET', 'POST'])
def company(name):
	if name == "1m" or name == "2m" or name == "5m" or name == "15m" or name == "30m" or name == "60m" or name == "90m" or name == "1d" or name == "5d" or name == "1wk" or name == "1mo" or name == "3mo":
		return redirect(url_for('interval', name=name))
	con = sql.connect("company_list.db")
	con.row_factory = sql.Row
	curr = con.cursor()
	curr.execute("select symbol from NSE_list where company_name=?", (name,))
	rows = curr.fetchall()
	if len(rows) > 0:
		symbol = rows[0][0]
		symbol += ".NS"
		company.company_name = name
		company.symbol = symbol
	# 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
	tata_power = yf(company.symbol, result_range='60d', interval='1d', dropna='True').result
	df = pd.DataFrame(tata_power)
	filename = company.company_name
	filename += ".csv"
	df.to_csv('./stock_dataset/' + filename)
	dataset = pd.read_csv('./stock_dataset/' + filename, sep=',', header=None)
	label = dataset.values[:,0]
	column = dataset.values[:,4]
	max_value, min_value, labels, close_price = get_data(label, column)
	title = company.company_name + " Stock Price in INR"
# 	labels, close_price = predict_future_stock(labels, close_price, rows[0][0], label, tata_power)
	return render_template('line_chart.html', title=title, max=max_value, min=min_value, labels=labels, values=close_price)
