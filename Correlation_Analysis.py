import pandas as pd
import seaborn as sns


data = pd.read_excel("Final.xlsx",sheetname="PAGEOUT")

#Factorise text columns to numbers for correlation ready
lpar_labels, lpar_unique = pd.factorize(data.lpar)
hmc_labels, hmc_unique = pd.factorize(data.hmc)
server_labels, server_unique = pd.factorize(data.server)

#Append the factorised columns to dataframe
data['lpar_fact'] = lpar_labels
data['hmc_fact'] = hmc_labels
data['server_fact'] = server_labels

data_needed = data[['lpar_fact','hmc_fact','server_fact','value']]

#Find correlation between all columns in dataframe
corr = data_needed.corr()
#Correlation is high if value is equal to 1 and very less if equals to 0
print(corr)

#To see the correlation matrix in heatmap
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)