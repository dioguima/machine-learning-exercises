from sklearn.linear_model import LinearRegression
from pandas import read_csv

bmi_life_data = read_csv('bmi_and_life_expectancy.csv')

x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]

bmi_life_model = LinearRegression()
bmi_life_model.fit(x_values, y_values)

print(bmi_life_model.predict(21.07931))