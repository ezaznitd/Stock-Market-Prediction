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
