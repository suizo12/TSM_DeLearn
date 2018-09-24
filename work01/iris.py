import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


df = pd.read_csv('iris.txt', sep='\t')
print(df.describe())
print(df['Species'].describe())
print(df['Species'].head(100))
df['Colors'] = '#0392cf'
df['Colors'].loc[df['Species'] == 'setosa'] = '#7bc043'
df['Colors'].loc[df['Species'] == 'versicolor'] = '#ee4035'

print(df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']].head())
scatter_matrix(df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']], color=df['Colors'])
plt.show()
