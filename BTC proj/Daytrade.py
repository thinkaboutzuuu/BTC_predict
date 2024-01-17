from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.layers import Dropout
import tensorflow as tf
from keras.models import load_model
import pandas_ta as ta


 
# mse 1089968.5875982451 after scaling back
model = load_model('daily30-0_model.h5')


def pred(dataset):
    latest_data = dataset[-31:-1]  # 获取最后31行数据
    scal = MinMaxScaler(feature_range=(0,1))
    latest_data_scaled = scal.fit_transform(latest_data)

    # 构建输入数据，这里需要注意维度匹配
    X_latest = np.array([latest_data_scaled])
    X_latest = np.reshape(X_latest, (1, 30, 8))  # 1个样本，9个时间步，每步8个特征

    # 使用模型进行预测
    predicted_price = model.predict(X_latest)

    # 反归一化预测结果以获取实际的价格
    predicted_price_original = scal.inverse_transform(np.hstack((predicted_price, np.zeros((predicted_price.shape[0], 7)))))[:, 0]

    # 输出预测的闭市价
    print(f"Predicted closing price for the next day: {predicted_price_original[0]}")

df = pd.read_csv('BTC-USDday.csv')
df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
df['RSI'] = ta.rsi(df['Close'], length=100)
df['EMAF'] = ta.ema(df['Close'], length=20)
df = df.drop(columns=['Volume'])
df = df.drop(columns=['Date'])

print(pred(df))
