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
pca=PCA()


#narrow to quantitative data
file = open("D206/churn/quantitative_vars.csv", "r")
quantvars = list(csv.reader(file, delimiter=","))[0]
file.close
churn=churn.filter(quantvars)

# Standard scalar

std_churn=scalar.fit_transform(churn)

# PCA

pca.fit(std_churn)

print(pca.explained_variance_ratio_)

