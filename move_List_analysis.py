import pandas as pd
#import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import scipy
move_list= pd.read_csv("C:\\Users\\Dell\\Downloads\\DigiDB_movelist.csv")
#first lest check the head of our data
#move_list.head()

# now lets look for null values
"""m2 = move_list.isnull()
for column in m2.columns.values.tolist():
    print(column)
    print (m2[column].value_counts())
    print("")    """

# lets change the power coulmn values to float
move_list['Power'].astype("float")


# normalizing power levels using max-min method
move_list['power_normalized']= move_list['Power'] / move_list['Power'].max()
#move_list['power_normalized'].head()


# lets plot the power level normalized
plt.hist(move_list["power_normalized"])
plt.xlabel("power normalized")
plt.ylabel("count")
plt.title("power bins")
plt.show()


# adding bins for data
bins = np.linspace(min(move_list["power_normalized"]), max(move_list["power_normalized"]), 4)
group_names = ['Low', 'Medium', 'High']
move_list['power-binned'] = pd.cut(move_list['power_normalized'], bins, labels=group_names, include_lowest=True )
move_list[['power_normalized','power-binned']].head(20)
move_list['power-binned'].value_counts()

#lets plot the bins
plt.bar(group_names, move_list["power-binned"].value_counts())
plt.xlabel("power")
plt.ylabel("count")
plt.title("power bins")
plt.show()

#making type and attribute dummy variables
dummy_2=pd.get_dummies(move_list['Type'])
#print(dummy_2.head())
move_list = pd.concat([move_list, dummy_2], axis=1)
dummy_3=pd.get_dummies(move_list['Attribute'])
move_list=pd.concat([move_list,dummy_3],axis=1)
#print(move_list.head())
#print(move_list[['SP Cost','power_normalized']].corr())
#plotting categorical data related to SP
sb.regplot(x="power_normalized", y="SP Cost", data=move_list)
plt.ylim(0,)
plt.show()

sb.boxplot(x="Type",y="SP Cost",data=move_list)
plt.show()
sb.boxplot(x="Attribute",y="SP Cost",data=move_list)
plt.show()

# checking value counts of attribute and type and putting them to dataframe
attr_count=move_list['Attribute'].value_counts().to_frame()
attr_count.rename(columns={'Attribute': 'value_counts'}, inplace=True)
attr_count.index.name = 'Attribute'
#print(attr_count)
type_count=move_list['Type'].value_counts().to_frame()
type_count.rename(columns={'Type': 'value_counts'}, inplace=True)
type_count.index.name = 'Type'
#print(type_count)
move_list_g1=move_list[['Type','Attribute','Power']]
move_list_g1=move_list_g1.groupby(['Type','Attribute'],as_index=False).mean()
grouped_pivot = move_list_g1.pivot(index='Attribute',columns='Type')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
print(grouped_pivot)
#heatmapping power to ,attributes , type
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

#getting the pearson_coef, p_value for the most susspected coulmns correlated to the power
pearson_coef, p_value = scipy.stats.pearsonr(move_list['Light'], move_list['Power'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#this means there is a modreate evidance that light type and power are correlated as p-value os less than 0.05
pearson_coef, p_value = scipy.stats.pearsonr(move_list['Physical'], move_list['Power'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#since p-value is smaller than 0.001 there is a strong evidence that phsical type attacks and power are correlated


# now our last step is to make an ANOVA table
move_list_g2=move_list[['Type','Attribute','Power']]
move_list_g2=move_list_g2.groupby(['Type'])
# ANOVA
f_val, p_val = scipy.stats.f_oneway(move_list_g2.get_group('Physical')['Power'], move_list_g2.get_group('Magic')['Power'],
                              move_list_g2.get_group('Direct')['Power'],move_list_g2.get_group('Fixed')['Power'],move_list_g2.get_group('Support')['Power'])

print("ANOVA results: F=", f_val, ", P =", p_val)

