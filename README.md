# stock_price
XGBoost model to predict stock price using Covid-19 data

## 문제
> ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/1.png)
> - 코로나 팬데믹 발생 기간 동안 주가 예측이 잘 되지 않음
> - 코로나가 호재로 작용한 넷플릭스, 아마존의 경우 어느정도 예측성능이 나오지만
> - 코로나가 악재로 작용한 AA, BP의 경우 예측성능이 상당히 저조한 모습을 확인할 수 있음

## 솔루션
> ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/2.png) ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/6.png)
> - 코로나 데이터를 feature로 활용하여 주가를 예측해보자!

## 시계열 분해
> ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/3.png) ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/7.png)
> - 일별 누적 확진자수와 일별 사망자수 데이터의 시계열 분해

## 나스닥, S&P 500 지수
> ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/8.png) ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/9.png)

## 예측 종목
> ```
> stock_code = ['NFLX', 'AMZN', 'AAL', 'BP']
> ```
> - 호재기업 : 넷플릭스, 아마존
> - 악재기업 : AA, BP

## 주식 종가의 분해 시계열
>```
>  decomposition = seasonal_decompose(stock_data['Close'], model='multiplicative',period=20) 
>  stock_data['Close_trend'] = decomposition.trend.fillna(method = 'ffill').fillna(method = 'bfill') # 종가 추세 
>  stock_data['Close_seosonal'] = decomposition.seasonal.fillna(method = 'ffill').fillna(method = 'bfill') # 종가 계절성 
>  stock_data['Close_resid'] = decomposition.resid.fillna(method = 'ffill').fillna(method = 'bfill') # 종가 잔차
>```

## MA(이동평균), RSI(상대강도지수)
>```
>  for n in [14, 30, 50, 200]:
>      stock_data['ma' + str(n)] = talib.SMA(np.array(list(map(float,stock_data['Close'].values))), timeperiod=n)
>      stock_data['rsi' + str(n)] = talib.RSI(np.array(list(map(float,stock_data['Close'].values))), timeperiod=n)
>      feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]
>```

## XGBoost 모델
>```
>  xgb = XGBRegressor(n_estimators= 10000, learning_rate = 0.05, tree_method = 'gpu_hist')
>  xgb.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], early_stopping_rounds = 100, eval_metric = 'mae', verbose = 100)
>```

## 예측 결과
> ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/10.png) ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/11.png)
> ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/12.png) ![image](https://github.com/jungsungmoon/stock_price/blob/main/pic/%EC%83%88%20%ED%8F%B4%EB%8D%94/13.png)

