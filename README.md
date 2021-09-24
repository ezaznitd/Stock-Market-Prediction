# Stock Market Live Update and Future Stock Value
‘A rising stock market is usually aligned with a growing economy and leads to greater investor confidence’. Investors follow some investment strategy before investing his/her stock into the market. In that scenario previous stock history for any company is the key factor of investment. Futures and derivatives help increase the efficiency of the underlying market. In Indian market investors preffer ‘NSE’ companies because it provides more liquidity for its stocks.

## Solution:
1. Every ‘NSE’ company provides it’s historical stock data.
2. For any interval (i.e. 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1w, 1mo, 3mo, YTD etc) historical data can be extracted through ‘Yahoo Finance Api’.
3. ‘LSTM (Long Short Term Memory)’ neural network model can predict future stock price based on historical data for any company.
4. Social influencing can also affect stock market (i.e. relevant news article can increase stock value for a company or vice-versa)[4].
5. This methodology can be used to build a stock market website which will show historical stock data as well as future stock for any ‘NSE’ company.

## Methodology:
1. Every company provides it’s historical data. We can extract historical data for any interval for a particular company using ‘Yahoo Finance Api’. On the basis of historical data we can predict stock data for following data.
2. Our stock market website will always train prediction model on the basis of historical data and sentimental analysis of news article for all the comapnies and show results in web browser on the basis of user’s query.
3. As training Machine Learning model takes long time to execute so we divide our model into two part (website manipulation and training prediction model). Both website manipulation and training prediction model will run parallally.

## Background Process Scheduling:
- Our stock market website is build using ‘Flask’.
- Training any machine learning algorithm is a lots of time consuming process. So we have scheduled model training in the background in our website i.e. both website and model training will be running parallally so that model traing wouldn’t wait for any request. It will automatically update the trained model after a fixed interval.

### Pseudocode:
- app = Flask(__name__)
- scheduler = APScheduler()
- scheduler.init_app(app)
- scheduler.start()
- app.apscheduler.add_job(func=scheduled_task,trigger='interval',weeks=1,args=[100],id="background_process”)

## Sentimental Analysis:
‘Stock exchange’ is a subject that is highly affected by economic, social, and political factors. There are several factors e.g. external factors or internal factors which can affect and move the stock market. Stock prices rise and fall every second due to variations in supply and demand. Various Data mining techniques are frequently involved to solve this problem. But technique using machine learning will give more accurate, precise and simple way to solve such issues related to stock and market prices.

## Some overview of my stock market prediction website:
![GitHub Logo](/images/home.png)
![GitHub Logo](/images/company_list.png)
![GitHub Logo](/images/live_stock.png)

## Usage:
Stock Market Website is currently deployed in Heroku. The website can be accessed through https://stockvalueprediction.herokuapp.com/

# Stock Market Live Update and Future Stock Value
Using this repository one can find live stock data of any company under `NSE` as well as can check what
will the future stock value for that particular company.

## Installation
1. Install [Python 3](https://www.python.org/) in your system. Make sure that `pip` is available for installing Python packages.
2. Install [`virtualenv`](https://virtualenv.pypa.io/en/latest/) for creating Python Virtual Environments.
    ```bash
    pip install virtualenv
    ```
3. Clone this repository or Extract it into a specific folder and `cd` into it.
    ```bash
    cd StockMarket-website
    ```
4. Create vitual environment called `env` using `virtualenv`.
    - Linux  or Mac
        ```bash
        virtualenv env
        source env/bin/activate
        ```
    - Windows
        ```
        virtualenv env
        env\Scripts\activate
        ```
    You can use the `deactivate` command for deactivating the virtual environment.
5. Install the required Python packages by the command:
    ```bash
    pip install -r requirements.txt
    ```


## Usage
1. You can start the Flask app server by calling
   ```bash
   python3 run.py
   ```
2. The website can be accessed through a browser at [127.0.0.1:5000](http://127.0.0.1:5000/) or [localhost:5000](localhost:5000)
