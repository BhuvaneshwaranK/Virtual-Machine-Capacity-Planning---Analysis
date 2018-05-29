import pandas as pd
import seaborn as sns


data = pd.read_excel("Final.xlsx",sheetname="Combined")

#Doing one-hot encoding (Converting row values to column values)
data = pd.get_dummies(data, columns=['metric'])

#Factorise text columns to numbers for correlation ready
data['hmc'],unique = pd.factorize(data['hmc'])
data['server'],unique = pd.factorize(data['server'])
data['lpar'],unique = pd.factorize(data['lpar'])

data_needed = data.drop(['rpttime'],axis=1)

#Find correlation between all columns in dataframe
corr = data_needed.corr()
#Correlation is high if value is equal to 1 and very less if equals to 0
print(corr)

#To see the correlation matrix in heatmap
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)