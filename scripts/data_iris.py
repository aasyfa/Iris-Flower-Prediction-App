from sklearn.datasets import load_iris
import pandas as pd

# Load this iris dataset as your dataframe, then export as csv to data folder
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('data/iris.csv', index=False)
print("Iris dataset saved to data/iris.csv")