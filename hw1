- 👋 Hi, I’m @yalindenizt
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
yalindenizt/yalindenizt is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

price = pd.read_csv(r"C:\Users\yalin.tektas\Desktop\PatikaVeriSetleri\ev_fiyat_tahmini.csv", sep=";")
print(price)
price_new = price.drop(["index", "oda_sayısı"], axis = 1)
print(price_new)
x = price_new["MetreKare"]
y = price_new["fiyatlar"]
plt.scatter(x,y)
plt.xlabel("MetreKare")
plt.ylabel("fiyatlar")
plt.show()
sns.jointplot(x = "MetreKare", y = "fiyatlar", data = price_new, kind = "reg")
x_ = price_new[["MetreKare"]]
y_ = price_new[["fiyatlar"]]
reg = LinearRegression()
model = reg.fit(x_, y_)
print(model)
dir(model)
intercept_model = model.intercept_
print("interception:", intercept_model)
slope_model = model.coef_
print("slope:", slope_model)
score_model = model.score(x_, y_)
print("ratio:", score_model)
y_prediction = model.predict(x_)
estimated_price = np.array([[150]])
prediction = model.predict(estimated_price)
print("estimated_price:", prediction)
print("MAE", mean_absolute_error(y, y_prediction))
print("MSE", mean_squared_error(y, y_prediction))
print("R^2:", r2_score(y, y_prediction))
print("RMSE:", sqrt(mean_squared_error(y, y_prediction)))

linearModelStats = sm.OLS(y_, x_)
modelStats = linearModelStats.fit()
print("STATS Model Values:", modelStats.summary())
print("estimated_price:", model.predict([[150]]))
