import numpy as np
import pandas as pd


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Load only the first 4 columns as floating point numbers
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# 1.⁠ ⁠Define two custom numpy arrays, say A and B.
# Generate two new numpy arrays by stacking A and B vertically and horizontally.

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

vertical_stack = np.vstack((A, B))
print("Vertically Stacked:\n", vertical_stack)

horizontal_stack = np.hstack((A, B))
print("Horizontally Stacked:\n", horizontal_stack)


#2.⁠ ⁠Find common elements between A and B. [Hint : Intersection of two sets]

common_elements = np.intersect1d(A, B)
print("Common Elements:", common_elements)


#3.⁠ ⁠Extract all numbers from A which are within a specific range.
# eg between 5 and 10. [Hint: np.where() might be useful or boolean masks]

filtered_A = A[(A >= 5) & (A <= 10)]
print("Numbers in A between 5 and 10:", filtered_A)


#4.⁠ ⁠Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0

filtered_iris = iris_2d[(iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)]
print("Filtered Rows:\n", filtered_iris)







#1.⁠ ⁠From df filter the 'Manufacturer', 'Model' and 'Type' for every 20th row starting from 1st (row 0).

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

filtered_df = df.loc[::20, ['Manufacturer', 'Model', 'Type']]
print(filtered_df)

#2.⁠ ⁠Replace missing values in Min.Price and Max.Price columns with their respective mean.

# Replace missing values with column mean
df['Min.Price'] = df['Min.Price'].fillna(df['Min.Price'].mean())
df['Max.Price'] = df['Max.Price'].fillna(df['Max.Price'].mean())

# Check if missing values are replaced
print(df[['Min.Price', 'Max.Price']].isnull().sum())

#3.⁠ ⁠How to get the rows of a dataframe with row sum > 100?

df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))

filtered_rows = df[df.sum(axis=1) > 100]
print(filtered_rows)