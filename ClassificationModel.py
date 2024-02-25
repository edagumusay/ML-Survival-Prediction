import pandas as pd
import numpy as np

df1 =  pd.read_csv('ttrain.csv')
df2 = pd.read_csv('ttest.csv')

df = pd.concat([df1,df2])

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0]) # filling missing values in the 'Embarked' column with 'S' (mode)

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
df['Title'] = df['Title'].replace(['Ms', 'Mlle'], 'Miss')
df['Title'] = df['Title'].replace(['Mme', 'Countess', 'Lady', 'Dona'], 'Mrs')  # assuming these titles are typically used by married women
df['Title'] = df['Title'].replace(['Dr', 'Major', 'Col', 'Sir', 'Rev', 'Jonkheer', 'Capt', 'Don'], 'Mr')  # assuming these titles are typically used by men

df['Age'].fillna(df.groupby('Title')['Age'].transform('median'), inplace=True)  # filling missing values in the 'Age' column with the median age based on the title

df['Fare'] = df['Fare'].fillna(df['Fare'].mode()[0])  # filling missing values in the 'Fare' column with the mode (most frequent value) of the fare

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)  # dropping unnecessary columns

df = pd.get_dummies(df, drop_first=True)  # converting categorical variables into binary indicators

train = df[:891]  # splitting the DataFrame into training and testing sets
test1 = df[891:]

x = train.drop(['Survived'], axis=1)
y = train['Survived']

test = test1.drop('Survived', axis=1)


## Modelling

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# initializing models
g = GaussianNB()
b = BernoulliNB()
r = RandomForestClassifier()
gr = GradientBoostingClassifier()

# fitting models
g.fit(x, y)
b.fit(x, y)
r.fit(x, y)
gr.fit(x, y)

# making predictions
gprediction = g.predict(test)
bprediction = b.predict(test)
rprediction = r.predict(test)
grprediction = gr.predict(test)

# creating DataFrames for predictions
prediction1 = pd.DataFrame({'PassengerId': df2['PassengerId'], 'Survived': gprediction.astype(int)})
prediction2 = pd.DataFrame({'PassengerId': df2['PassengerId'], 'Survived': bprediction.astype(int)})
prediction3 = pd.DataFrame({'PassengerId': df2['PassengerId'], 'Survived': rprediction.astype(int)})
prediction4 = pd.DataFrame({'PassengerId': df2['PassengerId'], 'Survived': grprediction.astype(int)})

# writing predictions to CSV files
prediction1.to_csv('prediction1.csv', index=False)
prediction2.to_csv('prediction2.csv', index=False)
prediction3.to_csv('prediction3.csv', index=False)
prediction4.to_csv('prediction4.csv', index=False)