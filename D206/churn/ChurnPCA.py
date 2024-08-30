import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#requirements
np.set_printoptions(legacy='1.25')
#Load Data
churn = pd.read_csv('D206/churn/churn_clean.csv')

scalar=StandardScaler()
pca=PCA(n_components=.7, svd_solver='full')


#narrow to quantitative data
file = open("D206/churn/quantitative_vars.csv", "r")
quantvars = list(csv.reader(file, delimiter=","))[0]
file.close
churn=churn.filter(quantvars)
#get column names
feature_names=churn.columns
#create PCA
churn_std=scalar.fit_transform(churn)
churn_PCA=pca.fit_transform(churn_std)
#Create PCA dictionary with comonent names [In-Text Citation:  (Recovering Feature Names of explained_variance_ratio_ in PCA with sklearn, 2024)]
component_weights=pca.components_
feature_weights_mapping = {}
for i, component in enumerate(component_weights):
  component_feature_weights = zip(feature_names, component)
  sorted_feature_weight = sorted(
      component_feature_weights, key=lambda x: abs(x[1]), reverse=True)
  feature_weights_mapping[f"Component {i+1}"] = sorted_feature_weight
mapping=pd.DataFrame.from_dict(feature_weights_mapping)
mapping.to_csv('D206/churn/feature_weights_mapping.csv')
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(loadings, columns=mapping.columns, index=churn.columns)
loadings_df.to_csv('D206/churn/loadings.csv')
print(pca.explained_variance_ratio_)
print("Feature names contributing to Principal Components")
for feature, weight in feature_weights_mapping.items():
  print(f"{feature}: {weight}") 
scree = np.arange(pca.n_components_) + 1
with PdfPages('D206/churn/ChurnPCA.pdf') as pp:
    plt.plot(scree, pca.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    pp.savefig()
    pp.close
plt.show()
