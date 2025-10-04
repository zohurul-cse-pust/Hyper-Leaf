import pandas as pd  # Import pandas library for data handling

df = pd.read_csv("train.csv")  # Load the CSV file into a DataFrame

X = df[["GrainWeight", "Gsw", "PhiPS2", "Fertilizer"]]  # Select feature columns for model input

y = df[["Heerup", "Kvium", "Rembrandt", "Sheriff"]]  # Select label columns (target variables)

print("Features (X):")  # Print title for features
print(X.head())  # Display first 5 rows of the features

print("\nLabels (y):")  # Print title for labels
print(y.head())  # Display first 5 rows of the labels
