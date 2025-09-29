import numpy as np

# it fixed data for every execution.
np.random.seed(42)

# Create realistic ecommerce dataset
n_product = 200

# Generate features
product_categories = np.random.choice(['Electronic','Clothing','Home & Kitchen','Book','Beauty'],n_product)
price = np.random.randint(10,500,n_product)
discount = np.random.uniform(0,0.5,n_product)
customer_rating = np.random.uniform(3.0,5.0,n_product)
review_count = np.random.randint(5,500,n_product)
ad_spend = np.random.uniform(5,500,n_product)
season = np.random.choice(['Spring','Summer','Fall','Winter'],n_product)

# Generate sales based on feature
base_sales = 1000
sales = (base_sales + (price * -2) + (discount * 3000)+ (customer_rating * 200)+(review_count * 0.5)+(ad_spend*0.8)+np.random.normal(0,200,n_product))
         

# Add seasonal effects
season_effect = {'Spring':1000,'Summer':300,'Fall':200,'Winter':400}
for i,s in enumerate(season):
    sales[i] +=season_effect[s]

# Add category effects
category_effect = {
        'Electronic':500,
        'Clothing':300,
        'Home & Kitchen':200,
        'Book':100,
        'Beauty':400
        }

for index in range(n_product):
    sales[i] += category_effect[product_categories[i]]

# Ensure no negative sales
sales = np.maximum(sales,0)

# Create a DataFrame
import pandas as pd

data = {
        'Product Categories':product_categories,
        'Price':price,
        'Discount':discount,
        'Customer Rating':customer_rating,
        'Review Count':review_count,
        'Ad Spend':ad_spend,
        'Season':season,
        'Monthly Sales':sales
       }
df = pd.DataFrame(data)

print("\n===DISPLAY TOP 15 RECORD===")
from tabulate import tabulate
print(tabulate(df.head(15),headers='keys',tablefmt='psql',showindex=False))
print(f"Dataset shape: {df.shape}")

print("\n===DATA EXPLORATION===")
print(f"Total product:{df.shape[0]}")
print(f"Total feature:{df.shape[1]-1}")
print(f"Target variable: 'monthly sales'")
print(f"Memory uses: {df.memory_usage(deep=True).sum() /1024:.1f}kb")

print("\n2 DATA TYPES AND MISSING VALUES:") 
print(df.info())
print(df.describe())

