from tabulate import tabulate

# Create sample pizza data
data = {
        'size_inches':[10,12,14,16,8,18,20,12,14,10],
        'toppings':[2,3,4,5,1,6,6,2,3,4],
        'price':[12,18,25,30,8,35,40,15,20,16]
       }
# Create DataFrame of pizza data
import pandas as pd
df = pd.DataFrame(data)
print(tabulate(df,headers='keys',tablefmt='psql',showindex=True))

# Seperate Features and Target
feature = df.drop('price',axis=1)#what we use to predict
target = df['price'] # what we want to predict

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(feature,target)


#Get coeficient and intercept
coefficient = model.coef_# per inch how much need to add price.
intercept = model.intercept_ #base price

print("\n===MODEL COEFFICIENT===")
print(f"intercept (BASE VALUE): {intercept:.2f}")
print(f"coefficient per pizaa inch: {coefficient[0]:.2f}")
print(f"coefficient per toppings: {coefficient[1]:.2f}")

print(f"\nFormula price: {intercept:.2f} + ({coefficient[0]:.2f} * size) + ({coefficient[1]:.2f} * toppings)")

# Example 14 inch pizza with 3 toppings
size = 14
toppings = 3

print("\n===MANUAL PREDICTION===")
predict_price = intercept + (coefficient[0]*size) + (coefficient[1]*toppings)
print(f"predict_price ${predict_price}")

# Lets check with model 
y_predict = model.predict([[size,toppings]])
print(f"model predicted: {y_predict[0]}")


#print(tabulate(target,headers='keys',tablefmt='psql',showindex=True))
