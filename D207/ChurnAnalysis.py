#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from seaborn import objects as so
from scipy.stats import f_oneway
from scipy.stats import pearsonr
from scipy.stats import cramervonmises_2samp as cv2s


np.set_printoptions(legacy='1.25')

#Load data
churn = pd.read_csv('D207/churn_clean.csv')
#Code for Rubric B1
#Get Columns by type
age_pay=churn.groupby("PaymentMethod")["Age"].apply(list)


#Run Initial Analysis
with PdfPages('D207/Output/D207.pdf') as pp:
    x0, y0=["Age", "PaymentMethod"]
    plt.style.use('fivethirtyeight')
    anova=f_oneway(age_pay.iloc[0],age_pay.iloc[1],age_pay.iloc[2],age_pay.iloc[3])
    fig, ax = plt.subplots(figsize=(25, 10))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0, hspace=0)
    ax.margins(1,.1)
    ax = sns.boxplot(churn, x=x0, y=y0)
    ax.set_xlabel(x0 + '\n1.5 X IQR')
    ax.set_ylabel("")
    ax.set_xlim(churn[x0].min()-churn[x0].max()*0.1, churn[x0].max()*1.1)
    sns.despine(trim=True, left=True)
    #sns.stripplot(churn, x=x0, y=y0, size=4, color=".3")
    fig.text(0.5, 0.95, 'ANOVA output for coorelation of payment method and customer age:\n' + str(anova), size = 12, horizontalalignment='center', verticalalignment='top')
    pp.savefig()
    
#Code for Rubric C part 1
    quantvars1=["MonthlyCharge", "Bandwidth_GB_Year"]
    quant=churn.filter(quantvars1)
    for i, col in enumerate(quantvars1):
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        fig.suptitle(col, fontsize=14, fontweight='bold')
        ax = sns.boxplot(y=quant[col], ax=axes.flatten()[0])
        ax.margins(1,.1)
        ax.text(ax.get_xlim()[0], ax.get_ylim()[1], quant[col].describe().round(2), fontsize=15, verticalalignment='top')
        ax.set_ylim(quant[col].min()-quant[col].max()*0.1, quant[col].max()*1.1)
        ax.set_ylabel("")
        ax.set_xlabel(col + '\n1.5 X IQR')
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        ax = sns.histplot(quant[col], ax=axes.flatten()[1])
        pp.savefig()
    #Code for Rubric C1
    qualvars1=["Contract","Marital"]
    qual=churn.filter(qualvars1)
    for i, col in enumerate(qualvars1):
        fig, axes = plt.subplots(1, 1, figsize=(25, 10))
        plt.subplots_adjust(left=.2, right=.8, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        fig.suptitle(col, fontsize=14, fontweight='bold')
        ax = sns.countplot(x=churn[col],order=qual[col].value_counts(ascending=False).index)
        abs_vals= qual[col].value_counts(ascending=False).values
        pct_val= qual[col].value_counts(ascending=False, normalize=True).values*100
        bar_lbl=[f'{p[0]} ({p[1]:.2f}%)' for p in zip(abs_vals, pct_val)]
        ax.bar_label(container=ax.containers[0], padding=-32, color='white', fontweight='bold', labels=bar_lbl )
        ax.margins(.1,.2)
        ax.text(ax.get_xlim()[0], ax.get_ylim()[1], qual[col].describe().round(2), fontsize=15, verticalalignment='top')
        pp.savefig()

#Code for Rubric D1
    quantvars2=["Population", "Outage_sec_perweek"]
    x1 ,y1=[quantvars2[0],quantvars2[1]]
    quant=churn.filter(quantvars2)
    Pearson= pearsonr(quant[x1], quant[y1])
    fig, axes = plt.subplots(1, 1, figsize=(25,25))
    fig.suptitle(x1+' vs '+y1, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.9, str(Pearson), size = 20, horizontalalignment='center', verticalalignment='top')
    ax = sns.scatterplot(quant,x=x1,y=y1)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    pp.savefig()
#Code for Rubric D2
    qualvars2=["Marital", "Children"]
    x1 ,y1=[qualvars2[0],qualvars2[1]]
    qual=churn.filter(qualvars2).astype("category")
    qual[x1]=qual[x1].cat.reorder_categories(['Married', 'Never Married', 'Separated', 'Widowed', 'Divorced'], ordered=True)
    qual[y1]=qual[y1].cat.reorder_categories([10,9,8,7,6,5,4,3,2,1,0], ordered=True)
    cramerv= cv2s(qual[x1].apply(str), qual[y1].apply(str))
    fig, axes = plt.subplots(1, 1, figsize=(25,25))
    fig.suptitle(x1+' vs '+y1, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.95, str(cramerv), size = 20, horizontalalignment='center', verticalalignment='top')
    ax = sns.histplot(qual,x=x1,hue=y1, multiple="stack", palette="winter")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    pp.savefig()
