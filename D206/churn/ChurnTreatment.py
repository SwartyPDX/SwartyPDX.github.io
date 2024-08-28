import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import csv

#Load Data
churnClean = pd.read_csv('D206/churn/churn_raw_data.csv')
#Remove rows with more than 'n' missing variables
n=2
churnClean=churnClean.dropna(thresh=(len(churnClean.columns)-n))
print(churnClean.info(verbose=False))
#Treatment for outliers in quantitative values
file=open("D206/churn/quantitative_vars.csv", "r")
quantvars=list(csv.reader(file, delimiter=","))[0]
file.close
quantChurn=churnClean.filter(quantvars)
#remove rows with negative and NA values in quantative columns
for i, col in enumerate(quantChurn):  
    churnClean=churnClean.drop(churnClean.index[quantChurn[col]<0])
    quantChurn=churnClean.filter(quantvars)
    
churnClean=churnClean.dropna(subset=quantvars)


print(churnClean.info(verbose=False))
churnClean.to_csv('D206/churn/churn_clean.csv', index=False)
