#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as PP
import seaborn as sns
import csv
from scipy.stats import f_oneway
import pingouin as pg

np.set_printoptions(legacy='1.25')

#Load data
churn = pd.read_csv('D207/churn_clean.csv')
#Code for Rubric B1
#Get Columns by type
age_pay=churn.groupby("PaymentMethod")["Age"].apply(list)

#Run Initial Analysis
anova=f_oneway(age_pay.iloc[0],age_pay.iloc[1],age_pay.iloc[2],age_pay.iloc[3])
print(anova)


#Code for Rubric C and C1




#Code for Rubric D1 and D2