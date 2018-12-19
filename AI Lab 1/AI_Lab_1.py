from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')

#example 1
#survided_to_Pclass = data.pivot_table('PassengerId', 'Pclass', 'Survived', 'count')
#print(survided_to_Pclass)

#survided_to_Pclass.plot(kind='bar', stacked=False)
#plt.show()
#end of example 1

#example 2
#fig, axes = plt.subplots(ncols=2) 

#suvived_to_SibSp = data.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count')
#suvived_to_SibSp.plot(ax=axes[0], title='SibSp') 

#suvived_to_Parch = data.pivot_table('PassengerId', ['Parch'], 'Survived', 'count')
#suvived_to_Parch.plot(ax=axes[1], title='Parch')

#plt.show()
#end of example 2

#example 3
#data.Embarked[data.Embarked.isnull()] = EmbarkedPassengerCount[EmbarkedPassengerCount == EmbarkedPassengerCount.max()].index[0]

#data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1) 
#print(data)
#end of example 3

#variant 1 task
#print('Count male passengers in 1st class:')
#Pclass_sex_pivotTable= data.pivot_table('PassengerId', 'Pclass', 'Sex', 'count')
#print(Pclass_sex_pivotTable.loc[1, 'male'])

#grouping_by_Pclass_sex = data.groupby(['Pclass','Sex'])['PassengerId'].count()
#print(grouping_by_Pclass_sex.loc[1, 'male'])
#end of variant 1 task

#variant 2 task
#print('Count children in 2nd class:')
#Pclass_age_pivotTable= data.pivot_table('PassengerId', 'Pclass', 'Age', 'count')
#print(Pclass_age_pivotTable.loc[2, 0 : 18].sum())

#grouping_by_Pclass_age = data.groupby(['Pclass','Age'])['PassengerId'].count()
#print(grouping_by_Pclass_age.loc[2].loc[0:18].sum())
#end of variant 2 task

#variant 3 task
print('Count lone passengers:')
grouping_by_SibSp_Parch = data.groupby(['SibSp','Parch'])['PassengerId'].count()
print(grouping_by_SibSp_Parch.loc[0,0])

pivot_table_SibSp_Parch = data.pivot_table('PassengerId', 'SibSp', 'Parch', 'count')
print(pivot_table_SibSp_Parch.loc[0,0])
#end of variant 3 task