import pandas as pd
df = pd.read_csv("train.csv")  
X = df[["GrainWeight", "Gsw", "PhiPS2", "Fertilizer"]]
y = df[["Heerup", "Kvium", "Rembrandt", "Sheriff"]]

print("Features (X):")
print(X.head())

print("\nLabels (y):")
print(y.head())