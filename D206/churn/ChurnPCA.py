import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#Load Data
cleanchurn = pd.read_csv('D206/churn/churn_clean.csv')


# Standard scalar



