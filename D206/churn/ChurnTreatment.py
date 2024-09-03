import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import csv

#Load Data with NaN dictionary to keep "none" option
NaNdict = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN","<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]
churnClean = pd.read_csv('D206/churn/churn_raw_data.csv', keep_default_na=False, na_values=NaNdict)


#Remove rows with NA
churnClean=churnClean.dropna()
print(churnClean.info(verbose=False))




#Treatment for outliers in quantitative values
file=open("D206/churn/quantitative_vars.csv", "r")
quantvars=list(csv.reader(file, delimiter=","))[0]
file.close
quantChurn=churnClean.filter(quantvars)

#Remove rows with NA
churnClean=churnClean.dropna()
print(churnClean.info(verbose=False))


#remove rows with negative values in quantative columns
for i, col in enumerate(quantChurn):  
    churnClean=churnClean.drop(churnClean.index[quantChurn[col]<0])
    quantChurn=churnClean.filter(quantvars)
    


print(churnClean.info(verbose=False))
churnClean.to_csv('D206/churn/churn_clean.csv', index=False)

#Check that data is clean

numunique=churnClean.nunique()
numduplicates=churnClean.value_counts('CaseOrder')
print(numduplicates)
print(numunique)
with PdfPages('D206/churn/Output/ChurnTreated.pdf') as pp:
    msno.bar(churnClean, figsize=(25, 20))
    plt.title('Bar Chart')
    pp.savefig()
    for i, col in enumerate(quantvars):
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        #Create subplot and use 1.5*IQR for outliers
        ax = sns.boxplot(y=quantChurn[col], ax=axes.flatten()[0])
        #Set Plot min and max
        ax.set_ylim(quantChurn[col].min()-quantChurn[col].max()*0.1, quantChurn[col].max()*1.1)
        #labels
        ax.set_ylabel("")
        ax.set_xlabel(col + '\n1.5 X IQR')
        #create distribution plot
        ax = sns.histplot(quantChurn[col], ax=axes.flatten()[1])
        pp.savefig()
    pp.close
