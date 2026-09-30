import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import(mean_absolute_error,mean_squared_error,r2_score)
data=pd.read_csv("c:\\Users\\Tejeshewini\\OneDrive\\Desktop\\excel data\\Data.csv")
print(data)
print(data.head())
print(data.tail())
print(data.columns)

#remove extra space from columns names
data.columns=data.columns.str.strip()
print(data.columns)

#duplicate rows
print(data.duplicated().sum())

#missing values
print(data.isnull().sum())

#data types
data["DATOP"]=pd.to_datetime(data["DATOP"],dayfirst=True,errors="coerce")
data["STD"]=pd.to_datetime(data["STD"],dayfirst=True,errors="coerce")

#remove the rows with invalid important dates
data=data.dropna(subset=["DATOP","STD","STA"])

#convert target to numeric
data["target"]=pd.to_numeric(data["target"],errors="coerce")

#remove rows without target
data=data.dropna(subset=["target"])

#remove impossible negative target values
data=data[data["target"]>=0]

#fearture engineering
data["dep_Hour"]=data["STD"].dt.hour
data["day_of_week"]=data["STD"].dt.dayofweek
data["month"]=data["STD"].dt.month
data["Day"]=data["STD"].dt.day

#remove Id
data=data.drop(columns=["ID"])

#status remove
data=data.drop(columns=["STATUS"])

#check data
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.dtypes)
print(data.info())
print(data.describe())

#check the  target flight delay
print(data["target"].describe())

#histogram vizulaization
plt.figure(figsize=(10,5))
sns.histplot(data["target"],bins=50,kde=True)
plt.title("FLIGHT DELAY DISTRIBUTION")
plt.xlabel("delay in minutes")
plt.ylabel("Number of flights")
plt.show()


#zero delaye flights
on_time=(data["target"]==0).sum()
delayed=(data["target"]>0).sum()

print("on-time flights:",on_time)
print("delayed flights:",delayed)

#in percentage
total=len(data)
print("on-time%:",round(on_time/total*100,2))
print("delayed%:",round(delayed/total*100,2))

#delay by departure
dep_delay=(data.groupby("DEPSTN")["target"].mean().sort_values(ascending=False))
print(dep_delay.head(10))

#plot vizualization
plt.figure(figsize=(10,5))
dep_delay.head(10).plot(kind="bar")
plt.title("top 10 departure station by avg delay")
plt.xlabel("departure station")
plt.ylabel("avg delay(minute)")
plt.show()

#delay by arrival station
arr_delay=data.groupby("ARRSTN")["target"].mean().sort_index(ascending=False)
print(arr_delay)

#plot
plt.figure(figsize=(10,5))
arr_delay.head(10).plot(kind="bar")
plt.title("top 10 arrival station by avg delay")
plt.xlabel("arrival station")
plt.ylabel("avg delay(minute)")
plt.show()

# delay by departure hour
hour_delay=(data.groupby("dep_Hour")["target"].mean())
print(hour_delay)

#plot
plt.figure(figsize=(10,5))
hour_delay.plot(kind="line",marker="o")
plt.title("avg flight delay by departure hour")
plt.xlabel("departure hour")
plt.ylabel("avg delay(minute)")
plt.grid()
plt.show()

#delay by month
month_delay=(data.groupby("month")["target"].mean())
print(month_delay)

#plot
plt.figure(figsize=(10,5))
month_delay.plot(kind="bar")
plt.title("avg flight delay by month")
plt.xlabel("month")
plt.ylabel("avg delay(minute)")
plt.show()


#delay by week
day_delay=(data.groupby("day_of_week")["target"].mean())
print(day_delay)

#plot
plt.figure(figsize=(8,5))
day_delay.plot(kind="bar")
plt.title("avg flight delay by day of week")
plt.xlabel("day of week")
plt.ylabel("avg delay(minute)")
plt.show()

#find the most delayed flights
print(data.sort_values("target",ascending=False)[["FLTID","DEPSTN","ARRSTN","target"]].head(10))

#aircraft analysis
aircraft_delay=(
    data.groupby("AC")["target"].agg(["count","mean"]).sort_values("mean",ascending=False))
print(aircraft_delay.head(10))

#feature engineering

data["dep_Hour"]=data["STD"].dt.hour
data["dep_minute"]=data["STD"].dt.minute
data["dep_weekday"]=data["STD"].dt.dayofweek
data["dep_month"]=data["STD"].dt.month
data["dep_Day"]=data["STD"].dt.day



#more understand
def get_time_period(hour):
    if hour<6:
        return "Early Morning"
    elif hour<12:
        return "Morning"
    elif hour<18:
        return "Afternoon"
    else:
        return "Evening"
data["Time_Period"]=data["dep_Hour"].apply(get_time_period)

#select feature and target
feature=["FLTID","DEPSTN","ARRSTN","AC","dep_Hour","day_of_week","month","Day"]
x=data[feature]
y=data["target"]

print(x.head())
print(y.head())
print(x.shape)
print(y.shape)

#separete categorical and numerical
categorical_feature=["DEPSTN","ARRSTN","AC"]
numerical_featiure=["dep_Hour","day_of_week","month","Day"]


# #split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#encode 
preprocessor=ColumnTransformer(transformers=[("cat",OneHotEncoder(handle_unknown="ignore"),categorical_feature),("num","passthrough",numerical_featiure)])

#random forest
rf_model=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
rf_pipline=Pipeline([("preprocessor",preprocessor),("model",rf_model)])
rf_pipline.fit(x_train,y_train)
print("training completed")
y_pred_rf=rf_pipline.predict(x_test)

mae=mean_absolute_error(y_test,y_pred_rf)
mse=mean_squared_error(y_test,y_pred_rf)
rmse=mse**0.5
r2=r2_score(y_test,y_pred_rf)
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2:",r2)

xgb_model=XGBRegressor(n_estimation=200,learning_rate=0.1,max_depth=6,random_state=42)
xgb_pipeline=Pipeline([("preprocessor",preprocessor),("model",xgb_model)])
xgb_pipeline.fit(x_train,y_train)
y_pred_xgb=xgb_pipeline.predict(x_test)

mae_xgb=mean_absolute_error(y_test,y_pred_xgb)
mse_xgb=mean_squared_error(y_test,y_pred_xgb)
rmse_xgb=mse_xgb**0.5
r2_xgb=r2_score(y_test,y_pred_xgb)
print("XGBoost MAE:",mae_xgb)
print("XgBoost MSE:",mse_xgb)
print("XGBoost RMSE:",rmse_xgb)
print("xgboost r2:",r2_xgb)

result=pd.DataFrame({"Model":["Random Forest","XGBoost"],
                     "MAE":[mae,mae_xgb],
                     "RMSE":[rmse,rmse_xgb],
                     "R2":[r2,r2_xgb]})
print(result)

joblib.dump(xgb_pipeline,"flight_delay_model.pkl")
print("model saved sucessufully")
model=joblib.load("flight_delay_model.pkl")
prediction=model.predict(x_test.iloc[[0]])
print("prediction delay:",prediction[0],"minutes")


