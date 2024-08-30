import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import csv
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
#Load Data
churn = pd.read_csv('D206/churn/churn_clean.csv')

scalar=StandardScaler()
pca=PCA(n_components=0.7, svd_solver='full')


#narrow to quantitative data
file = open("D206/churn/quantitative_vars.csv", "r")
quantvars = list(csv.reader(file, delimiter=","))[0]
file.close
churn=churn.filter(quantvars)
#get column names
feature_names=churn.columns
#create PCA
churn_scalar=scalar.fit_transform(churn)
churn_PCA=pca.fit(churn_scalar)
print(pca.fit_transform(churn_scalar))
#Create PCA dictionary with comonent names [In-Text Citation:(Recovering Feature Names of explained_variance_ratio_ in PCA with sklearn, 2024)]
component_weights=pca.components_
feature_weights_mapping = {}
for i, component in enumerate(component_weights):
  component_feature_weights = zip(feature_names, component)
  sorted_feature_weight = sorted(
      component_feature_weights, key=lambda x: abs(x[1]), reverse=True)
  feature_weights_mapping[f"Component {i+1}"] = sorted_feature_weight

print(pca.explained_variance_ratio_)
print("Feature names contributing to Principal Components")
for feature, weight in feature_weights_mapping.items():
  print(f"{feature}: {weight}") 

