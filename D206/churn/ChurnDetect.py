#py version 3.12
#import libraries
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

#Load Data
churn = pd.read_csv('D206\churn\churn_raw_data.csv')
#print(churn.head())

#duplicate
numunique=churn.nunique()

numduplicates=churn.value_counts("CaseOrder")
#ChurnDuplicates = churn.duplicated()

print(numduplicates)
print(numunique)

#missing - null, NA and ""
msno.bar(churn)
msno.matrix(churn, labels= True)
#plt.show()