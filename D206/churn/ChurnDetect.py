#py version 3.12
#import libraries

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import csv

#Load Data
churn = pd.read_csv('D206/churn/churn_raw_data.csv')

#Load the list of quantitative varibles
file = open("D206/churn/quantitative_vars.csv", "r")
quantvars = list(csv.reader(file, delimiter=","))[0]
file.close
quantChurn= churn.filter(quantvars)

pp = PdfPages('D206/churn/Churn_detect.pdf')
#find duplicates
numunique=churn.nunique()

numduplicates=churn.value_counts('CaseOrder')


print(numduplicates)
print(numunique)



#find missing - null, NA and ""
msno.bar(churn, figsize=(25, 20))
plt.title('Bar Chart')
pp.savefig()  # saves the current figure into a pdf page

msno.matrix(churn, labels= True, figsize=(25, 20))
plt.title('Matrix Chart')
pp.savefig()  # saves the current figure into a pdf page

msno.heatmap(churn, figsize=(25, 20))
plt.title('Heatmap Chart')
pp.savefig()  # saves the current figure into a pdf page




#Finding outliers and observing distrubution for anomalies
fig, axes = plt.subplots(2, len(quantvars), figsize=(25, 20))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.9, hspace=0.25)
for i, col in enumerate(quantvars):
    #Create subplot and use 1.5*IQR for outliers
    ax = sns.boxplot(y=quantChurn[col], ax=axes.flatten()[i])
    #Set Plot min and max
    ax.set_ylim(quantChurn[col].min()-quantChurn[col].max()*0.1, quantChurn[col].max()*1.1)
    #labels
    ax.set_ylabel("")
    ax.set_xlabel(col + '\n1.5 X IQR')
    #rotate to fit large numbers on axis
    if (quantChurn[col].max() > 999):
        ax.tick_params(axis='y', rotation=70)
for i, col in enumerate(quantvars):
    #Create subplot of histogram
    ax = sns.histplot(quantChurn[col], ax=axes.flatten()[i+len(quantvars)])
pp.savefig()  # saves the current figure into a pdf page
pp.close()



