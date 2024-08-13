#py version 3.12
#import libraries

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import csv

#Load Datacd
churn = pd.read_csv('churn_raw_data.csv')
print(*churn)
file = open("quantitative_vars.csv", "r")
quantvars = list(csv.reader(file, delimiter=","))[0]
file.close
print(quantvars)
quantChurn= churn.filter(quantvars)
print(quantChurn.head())
#duplicate
numunique=churn.nunique()

numduplicates=churn.value_counts("CaseOrder")
#ChurnDuplicates = churn.duplicated()

print(numduplicates)
print(numunique)

#missing - null, NA and ""
msno.bar(churn)
msno.matrix(churn, labels= True)

#Finding outliers
plt.figure()
#sns.boxplot(quantChurn)
fig, axes = plt.subplots(1, len(quantvars))
for i, col in enumerate(quantvars):
    ax = sns.boxplot(y=quantChurn[col], ax=axes.flatten()[i])
    ax.set_ylim(quantChurn[col].min(), quantChurn[col].max())
    ax.set_ylabel(col + ' / Unit')
plt.show()
plt.show()