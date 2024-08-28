import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import csv

#Load Data
churn_clean = pd.read_csv('D206/churn/churn_raw_data.csv')
#Remove rows with more than 'n' missing variables
n=2
churn_clean=churn_clean.dropna(thresh=(len(churn_clean.columns)-n))

#Treatment for outliers
#remove rows with negative values








print(churn_clean.info(verbose=False))
churn_clean.to_csv('D206/churn/churn_clean.csv', index=False)
