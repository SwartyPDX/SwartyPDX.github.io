#py version 3.12
#import libraries

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import csv

#Load Data with NaN dictionary to keep "none" option
NaNdict = [" ", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN","<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]
churn = pd.read_csv('D206/churn/churn_raw_data.csv', keep_default_na=False, na_values=NaNdict)

#Load the list of quantitative varibles and create quantative only dataframe
file = open("D206/churn/quantitative_vars.csv", "r")
quantvars = list(csv.reader(file, delimiter=","))[0]
file.close
quantChurn= churn.filter(quantvars)


#find duplicates
numunique=churn.nunique()

numduplicates=churn.value_counts('CaseOrder')


print(numduplicates)
print(numunique)


#find missing - null, NA and ""
with PdfPages('D206/churn/Output/ChurnDetect.pdf') as pp:
    msno.bar(churn, figsize=(25, 20))
    plt.title('Bar Chart')
    pp.savefig()  
    msno.matrix(churn, labels= True, figsize=(25, 20))
    plt.title('Matrix Chart')
    pp.savefig()  
    msno.heatmap(churn, figsize=(25, 20))
    plt.title('Heatmap Chart')
    pp.savefig()  
    for i, col in enumerate(quantvars):
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        ax = sns.boxplot(y=quantChurn[col], ax=axes.flatten()[0])
        ax.set_ylim(quantChurn[col].min()-quantChurn[col].max()*0.1, quantChurn[col].max()*1.1)
        ax.set_ylabel("")
        ax.set_xlabel(col + '\n1.5 X IQR')
        ax = sns.histplot(quantChurn[col], ax=axes.flatten()[1])
        pp.savefig()
 

#Output basic statistics of Quantitative data
quantChurn.describe().to_csv('D206/churn/Output/quantitativedetails.csv')

