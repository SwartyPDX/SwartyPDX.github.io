import pandas as pd
import numpy as np
import missingno as msno
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import csv

scaler=MinMaxScaler()

imputer=KNNImputer(n_neighbors=3)
#Load Data with NaN dictionary to keep "None" string from variables
NaNdict = [" ", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN","<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]
churnClean = pd.read_csv('D206/churn/churn_raw_data.csv', keep_default_na=False, na_values=NaNdict, index_col=0)
churn = pd.read_csv('D206/churn/churn_raw_data.csv', keep_default_na=False, na_values=NaNdict, index_col=0)
#Treatment for outliers in quantitative values
file=open("D206/churn/quantitative_vars.csv", "r")
quantvars=list(csv.reader(file, delimiter=","))[0]
file.close
quantChurn=churnClean.filter(quantvars)


numcols = quantChurn.select_dtypes(include=np.number).columns
#Change negative values to NaN in quantative columns
churnClean[numcols]=churnClean[numcols].mask(churnClean[numcols]<0)
print(churnClean[numcols].min())
quantChurn=churnClean.filter(quantvars)
qualChurn=churnClean.drop(quantvars, axis=1)
#Fit numeric rows with NaN by replacing with K Nearest Neighbors
quantChurnscale=pd.DataFrame(scaler.fit_transform(quantChurn), columns=quantvars) 

quantChurn = pd.DataFrame(imputer.fit_transform(quantChurnscale),columns = quantvars)
quantChurnfit=pd.DataFrame(scaler.inverse_transform(quantChurn), columns=quantvars)
Wholevaluecol=['Population','Children','Age','Email','Contacts','Yearly_equip_failure']
quantChurnfit[Wholevaluecol]=quantChurnfit[Wholevaluecol].round()
quantChurn=quantChurnfit.round(2)
churnClean[quantvars]=quantChurn[quantvars].values

# Fill in missing qualitative values
for column in churnClean.columns:
    churnClean[column].fillna(churnClean[column].mode()[0], inplace=True)




print(quantChurn.isna().any())
print(churnClean[quantvars].isna().any())

print(churnClean.info(verbose=False))
churnClean.to_csv('D206/churn/churn_clean.csv')

#Check that data is clean

numunique=churnClean.nunique()
numduplicates=churnClean.value_counts('CaseOrder')
print(numduplicates)
print(numunique)
quantChurn=churnClean.filter(quantvars)
quantchurnorig=churn.filter(quantvars)
with PdfPages('D206/churn/Output/ChurnTreated.pdf') as pp:
    msno.bar(churnClean, figsize=(25, 10))
    plt.title('Bar Chart')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.8, wspace=0.1, hspace=0.1)
    pp.savefig()
    for i, col in enumerate(quantvars):
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        ax = sns.histplot(quantchurnorig[col], ax=axes.flatten()[0])
        ax = sns.histplot(quantChurn[col], ax=axes.flatten()[1])
        pp.savefig()