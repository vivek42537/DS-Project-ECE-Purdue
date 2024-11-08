import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics as stat
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as MSE


data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected_2.csv', thousands=',')
data.colums = data.columns.str.strip()
print(data.columns)


data['Precipitation'] = data['Precipitation'].replace(['T'], '0')
data['Precipitation'] = data['Precipitation'].map(lambda x: x.rstrip(' (S)'))
# Precip = data['Precipitation']
# print(Precip)

data = data.set_index('Date')
print("PREEEEEE")
print(data)
# print("DAOIGHEONSOVGNSV:NDGVSNVGLN")
# print(data)
num_cols = ['Manhattan Bridge',
            'Williamsburg Bridge',
            'Queensboro Bridge',
            'Brooklyn Bridge']
data[num_cols] = data[num_cols].astype(float)
# plt.figure(figsize=(10,10))
data[num_cols].plot(figsize=(20,10), linewidth = 1)
#plt.figure(figsize=(10,10))
plt.xlabel('Date')
plt.ylabel('# of Bicycles')
plt.title('Bikes per Bridge')
plt.xticks(range(len(data.index)), data.index, rotation = 'vertical', fontsize = 'x-small')
plt.show()

data = data.reset_index()
print("POSTTTTTT")
print(data)

print("Brooklyn Bridge Standard Dev: ",stat.stdev(data['Brooklyn Bridge']), "Mean:", stat.mean(data['Brooklyn Bridge']))
print("Manhattan Bridge Standard Dev: ",stat.stdev(data['Manhattan Bridge']),"Mean:", stat.mean(data['Manhattan Bridge']))
print("Williamsburg Bridge Standard Dev: ",stat.stdev(data['Williamsburg Bridge']),"Mean:", stat.mean(data['Williamsburg Bridge']))
print("Queensboro Bridge Standard Dev: ",stat.stdev(data['Queensboro Bridge']),"Mean:", stat.mean(data['Queensboro Bridge']))

x = 0
for i in data['Brooklyn Bridge'] :
    if i >= stat.mean(data['Brooklyn Bridge']) - stat.stdev(data['Brooklyn Bridge']) and i <= stat.mean(data['Brooklyn Bridge']) + stat.stdev(data['Brooklyn Bridge']):
        x += 1
percStd = (x / len(data['Brooklyn Bridge'])) * 100
print('Percent within one StDev for Brooklyn:', percStd)

for i in data['Manhattan Bridge'] :
    if i >= stat.mean(data['Manhattan Bridge']) - stat.stdev(data['Manhattan Bridge']) and i <= stat.mean(data['Manhattan Bridge']) + stat.stdev(data['Manhattan Bridge']):
        x += 1
percStd = (x / len(data['Manhattan Bridge'])) * 100
print('Percent within one StDev for Manhattan:', percStd)

for i in data['Williamsburg Bridge'] :
    if i >= stat.mean(data['Williamsburg Bridge']) - stat.stdev(data['Williamsburg Bridge']) and i <= stat.mean(data['Williamsburg Bridge']) + stat.stdev(data['Williamsburg Bridge']):
        x += 1
percStd = (x / len(data['Williamsburg Bridge'])) * 100
print('Percent within one StDev for Williamsburg:', percStd)

for i in data['Queensboro Bridge'] :
    if i >= stat.mean(data['Queensboro Bridge']) - stat.stdev(data['Queensboro Bridge']) and i <= stat.mean(data['Queensboro Bridge']) + stat.stdev(data['Queensboro Bridge']):
        x += 1
percStd = (x / len(data['Queensboro Bridge'])) * 100
print('Percent within one StDev for Queensboro:', percStd)



# data['Total of Three'] = data['Manhattan Bridge'] + data['Queensboro Bridge'] + data['Brooklyn Bridge']

y = data['Total']
totReal = y
x1 = data['High Temp (°F)']
x2 = data['Low Temp (°F)']
plt.scatter(x1,y, c = 'r')
plt.xlabel('High Temp (°F)')
plt.ylabel('# of Bicycles')
plt.title('Bikes vs Temperature')
plt.show()
plt.scatter(x2,y)
plt.xlabel('Low Temp (°F)')
plt.ylabel('# of Bicycles')
plt.title('Bikes vs Temperature')
plt.show()

tot = y.sort_values()
# print(totThree)

print('MEAN:', stat.mean(tot))

topPerc = len(tot) * 0.065
# print(topPerc) #13.91

print('Mean of highest 6.5 percent:', stat.mean(tot[-14:]))
print('Mean of lowest 6.5 percent:', stat.mean(tot[0:14]))

arr = data.to_numpy()

xHigh = arr[:,2].reshape(-1,1)
xLow = arr[:,3].reshape(-1,1)

x = np.concatenate((xHigh, xLow), axis = 1)
x = np.array(x, dtype=float)
y = totReal.to_numpy()

print("SOOVHSOOOOOO")
print(x)

x9 = PolynomialFeatures(degree=2, include_bias = False).fit_transform(x)
model = LinearRegression().fit(x9, y)

rSq = model.score(x9, y)
print("R squared value:", rSq)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

x2 = sm.add_constant(x)
model = sm.OLS(y, x2)
results = model.fit()
print(results.summary())
print('OLS regression coefficients:', results.params)


yPre = data['Total']
xPre = data['Precipitation']
plt.figure(figsize=(15,10))
plt.xticks(rotation = 'vertical', fontsize = 'x-small')
plt.scatter(xPre,yPre, c = 'g')
plt.xlabel('Precipitation Level')
plt.ylabel('# of Bicycles')
plt.title('Bikes vs Precipitation')
plt.show()

x1 = arr[:,4].reshape((-1,1))
x1 = np.array(x1, dtype=float)
y1 = yPre.to_numpy()

x10 = PolynomialFeatures(degree=6, include_bias=False).fit_transform(x1)
model10 = LinearRegression().fit(x10, y1)

rSquare = model10.score(x10, y1)
print("R squared value Precipitation:", rSquare)
print("Intercept Precipitation:", model10.intercept_)
print("Coefficients Precipitation:", model10.coef_)

x11 = sm.add_constant(x1)
model11 = sm.OLS(y1, x11)
result = model11.fit()
print(result.summary())
print('OLS regression coefficients:', result.params)




xL = np.concatenate((xHigh, xLow, x1), axis = 1)
xL = np.array(xL, dtype=float)
y = totReal.to_numpy()

xXL = PolynomialFeatures(degree=2, include_bias = False).fit_transform(xL)
modelL = LinearRegression().fit(xXL, y)

rSqu = modelL.score(xXL, y)
print("R squared value three:", rSqu)
print("Intercept three:", modelL.intercept_)
print("Coefficients three:", modelL.coef_)

xS = sm.add_constant(xL)
model = sm.OLS(y, xS)
results = model.fit()
print(results.summary())
print('OLS regression coefficients three:', results.params)