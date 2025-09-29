import numpy as np
np.random.seed(42) # for reproducibility(fixed random numbers)
nums_cars = 200
# model_year = np.random.randint(2000,2024,nums_cars)
# mileage = np.random.randint(4000,10000,nums_cars)
# engine_size = np.random.randint(1.5,3.5, nums_cars)
# horsepower = np.random.randint(100,400,nums_cars)
# brands = np.random.choice(['Toyota','Ford','BMW','Audi','Mercedes'],nums_cars)
model_year = np.random.randint(2020,2023,nums_cars)
mileage = np.random.randint(5000,8000,nums_cars)
engine_size = np.random.uniform(1.0,3.5,nums_cars)
horsepower = np.random.randint(80,300,nums_cars)
brands = np.random.choice(['Toyota','Honda','Ford','BMW','Mercedes'],nums_cars)

base_price = 2000
price = (base_price + (2023 - model_year)* - 1500 + mileage * 0.15 + engine_size * 8000 + horsepower * 100 + np.random.randint(0,3000,nums_cars))
print("\n==== Sample Data ====")
print("DISPLAYING FIRST 10 ENTRIES")
print(f"{'Year':<11}{'Mileage':<10}{'Engine':<10}{'Horsepower':<12}{'Brands':<10}{'Price'}")
print("--"*60)
for i in range(10):
    y = model_year[i]
    m = mileage[i]
    e = engine_size[i]
    h = horsepower[i]
    b = brands[i]
    p = price[i]
    print(f"{y:<11}{m:<10}{e:<10.1f}{h:<12}{b:<10}{p:<10,.2f}")

# Brand premium more cost
brand_premium = {
    'Toyota':0,
    'Ford':2000,
    'BMW':8000,
    'Audi':1000,
    'Mercedes':10000
}
for i,brand in enumerate(brand_premium):
    price[i] +=brand_premium[brand]
    # print(f"{brand}: {brand_premium[brand]}")
    #enumerate given index and value of the list


# Create DataFrame
import pandas as pd
df = pd.DataFrame(
    {
        'Model Year': model_year,
        'Mileage': mileage,
        'Engine Size': engine_size,
        'Horsepower': horsepower,
        'Brand':brands,
        'Price':price
    }
)
print("\n==== DataFrame Info ====")
print("DISPLAYING DATAFRAME FIRST 5 ENTRIES")
from tabulate import tabulate
print(tabulate(df.head(5),headers='keys',tablefmt='psql',showindex=False))
print("\nBasic Statistics for Price Column")

print(f"Min Price: {df['Price'].min():,.2f}")
print(f"Max Price: {df['Price'].max():,.2f}")
print(f"Mean Price: {df['Price'].mean():,.2f}")
print(f"Median Price: {df['Price'].median():,.2f}")

print("\nFeature Statistics")
print(df[['Model Year','Mileage','Engine Size','Horsepower','Brand','Price']].describe())

#Preprocessing Data and handle categorical data
df_encoded = pd.get_dummies(df,columns=['Brand'],prefix='brand')
print(tabulate(df_encoded.head(10),headers='keys',tablefmt='psql',showindex=False))

# Split Features and Target
X = df_encoded.drop('Price',axis=1)
Y = df_encoded['Price']
print("\nFeatures and Target Shapes")
print(f"Features shape: {X.shape}")
print(f"Target shape: {Y.shape}")

# Split Data into Training and Testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print("\nTraning and Testing Shapes")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_test shape: {Y_test.shape}")

# Scale Features so that all features are on similar scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)
print("\nScaled Featuree Samples")
print(tabulate(x_train_scaled[:5],headers=X.columns,tablefmt='psql',showindex=False))

# Train Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
y_predict = model.predict(X_test)
print("\nModel Coefficients")
print("DISPLAYING MODEL COEFFICIENTS AND INTERCEPT")
coeff =model.coef_
intercept = model.intercept_
for i, c in enumerate(coeff):
    print(f"{X.columns[i]:<15}: {c:,.2f}")

print(f"{'Intercept':<15}: {intercept:,.2f}")

feature_important = pd.DataFrame({
    'feature':X.columns,
    'coefficent':coeff,
    'abs_coefficient':abs(coeff)
}).sort_values(by='abs_coefficient',ascending=False)
print("\nFeature Importance")
print(tabulate(feature_important,headers='keys',tablefmt='psql',showindex=False))

# Evaluate Model
print("\nPRINT 10 PREDICTION VS ACTUAL")
print(f"{'Actual':<15}{'Predicted':<15}{'Error':<15}{'Error %':<10}")
print("-"*60)
for i in range(10):
    actual = Y_test.iloc[i]
    predicted = y_predict[i]
    error = abs(actual - predicted)
    error_pct = (error / actual)*100
    # print(f"{actual:<15,.2f}{predicted:<15,.2f}{error:<15,.2f}{error_pct:<10.2f}")


# Iterate y_predict and Y_test
for actual,predicted in zip(Y_test,y_predict):
    error = abs(actual - predicted)
    error_pct = (error / actual)*100
    print(f"{actual:<15,.2f}{predicted:<15,.2f}{error:<15,.2f}{error_pct:<10,.2f}")

print(type(Y_test),':',type(y_predict))

# Convert actual and predicted into dataframe

result_df = pd.DataFrame({
    'Actual':Y_test,
    'Predicted':y_predict,
    'erro': abs(Y_test - y_predict),
    'error_pct':abs(Y_test - y_predict) / Y_test * 100
})
print("\nResult DataFrame Sample")
print(tabulate(result_df.head(5),headers='keys',tablefmt='psql',showindex=False))

print(f"Formula price = {intercept:,.2f}")
for i,row in feature_important.iterrows():
    print(f" + ({row['abs_coefficient']:,.2f} * {row['feature']}) +")


# Model Evaluation Metrics
print("\nModel Evaluation Metrics")
from sklearn.metrics import mean_absolute_error,mean_squared_error
mae =mean_absolute_error(Y_test,y_predict) # i.e average error in prediction
mse = mean_squared_error(Y_test,y_predict)
rmse = np.sqrt(mse)
r2 = model.score(X_test,Y_test)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Square Error {rmse}")
print(f"R^2 Score: {r2}")








