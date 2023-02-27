import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
from prophet.plot import plot_plotly

# using Streamlit to create a WebApp and functional sliders/buttons
# using YahooFinance to import our stock data
# using FacebookProphet to predict future stock prices
# using Plotly to create graphs from our data

# sets the start and end dates for the prediction
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# sets the title of the webapp
st.title("Stock Prediction App")

# asks the user to input a stock to add to the dataset
add_stock = st.text_input("Enter the name of a stock to add to the dataset")
# sets the stocks that will be predicted
stocks = ["PEP", "BBWI", "MSFT", "TSLA", add_stock]
# lets the user choose which stock they want to predict
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# creates a slider to let the user choose how many monthy  ahead they want to predict
n_months = st.slider("Months of prediction:", 1, 24)
# sets the period as the number of months times 30 (days in a month)
period = n_months * 30


# this will cache the data, so it won't need to be loaded again
@st.cache
# this function loads the data for the specified stock
def load_data(ticker):
    # downloads all the data from the starting date to the current date
    data = yf.download(ticker, START, TODAY)
    # this will put the date in the first column of the table
    data.reset_index(inplace=True)
    return data


# displays a loading message while the data is being loaded
data_load_state = st.text("Loading data...")
# loads the data for the selected stock
data = load_data(selected_stock)
# changes the message to say that the data has been loaded
data_load_state.text("Loading data...completed!")

# displays the raw data in a table
st.subheader('Raw data')
# shows the last 5 rows of the data
st.write(data.tail())


# this function plots the raw data for the stock
def plot_raw_data():
    # create a plotly graph object
    fig = go.Figure()
    # adds the stock's daily opening prices to the graph
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    # adds the stock's daily closing prices to the graph
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    # adds a slider to the graph to allow the user to zoom in or out on specific date ranges
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    # plots the graph
    st.plotly_chart(fig)


# calls the function to plot the raw data
plot_raw_data()

# forecasting data
# creates a dataframe to be used for training the model
df_train = data[['Date', 'Close']]
# renames the columns so that the FacebookProphet model can understand them
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# creates the FacebookProphet model
m = Prophet()
# fits the model with the training data
m.fit(df_train)
# creates a dataframe to hold future data
future = m.make_future_dataframe(periods=period)
# predicts the future data in the dataframe
forecast = m.predict(future)

# displays the forecast data in a table
st.subheader('Forecast data')
# shows the last 5 rows of the forecast data
st.write(forecast.tail())

# plots the forecast data using plotly
st.write('Forecast data')
# creates a plotly graph object using the model and forecast data
fig1 = plot_plotly(m, forecast)
# plots the graph object
st.plotly_chart(fig1)

# plots the forecast components
st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
