
# Commented out IPython magic to ensure Python compatibility.
from google.colab import files
uploaded = files.upload()

import pandas as pd

train_df = pd.read_csv("train.csv")
print(train_df)

print("count of null values:", train_df.isnull().sum())



#IMPUTATION!!!
from sklearn.impute import SimpleImputer
import numpy as np
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df['Cabin'] = train_df['Cabin'].fillna('Unknown')
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
train_df[train_df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(train_df.select_dtypes(include=[np.number]))


print("After Imputation!!!!!", train_df)


train_df = train_df.drop(columns = ['Cabin'])
train_df = train_df.drop(columns = ['Name'])
train_df = train_df.drop(columns = ['Ticket'])
train_df.columns

train_df['Sex'] = train_df['Sex'].map({'male' : 0, 'female' : 1})
print(train_df)                                                                           # "after removing the Cabin,Name,Ticket column beacause of a lot of missing values!!!"
train_df = pd.get_dummies(train_df, columns =['Embarked'])
print("encoded", train_df)



print("statistics:", train_df.describe())
print("information:",train_df.info())

import matplotlib
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
# %matplotlib inline


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] == 19
matplotlib.rcParams['figure.figsize']== (10,8)



# TO VISUALIZE THE DISTRIBUTION OF SURVIVED PEOPLE:
fig1 = px.histogram(train_df,
                    x = 'Survived',
                    nbins = 2,
                    marginal='box',
                    title ='DISTRIBUTION OF SURVIVED PEOPLE',
                    )
fig1.update_layout(width=900,
                  height=500,
                  bargap=0.2)
fig1.show()




# TO VISUALIZE THE DISTRIBUTION OF Pclass:
fig2 = px.histogram(train_df,
                    x = 'Pclass',
                    nbins = 3,
                    marginal='box',
                    title ='DISTRIBUTION OF Pclass',
                    )
fig2.update_layout(width=900,
                  height=500,
                  bargap=0.2)
fig2.show()


#TO VISUALIZE THE DISTRIBUTION OF PCLASS VS SURVIVED:
fig3 = px.histogram(train_df,
                    x = 'Pclass',
                    y = 'Survived',
                    nbins = 3,
                    marginal='box',
                    title ='DISTRIBUTION OF Pclass VS SURVIVED',
                    )
fig3.update_layout(width=900,
                  height=500,
                  bargap=0.2)
fig3.show()

#TO VISUALIZE THE DISTRIBUTION OF EMBARKED VS SURVIVED:
fig4 = px.histogram(train_df,
                    x = 'Embarked_S',
                    y = 'Survived',
                    nbins = 3,
                    marginal='box',
                    title ='DISTRIBUTION OF Pclass VS SURVIVED',
                    )
fig4.update_layout(width=900,
                  height=500,
                  bargap=0.2)
fig4.show()





print(train_df.corr())                                                                     #"correlation between different features!!!!"
#TRAIN TEST SPLIT
X = train_df.drop(columns=['Survived'])
y = train_df['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#SCALING
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#TRAIN MODEL

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)


#TEST MODEL

y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred)


#ACCURACY

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# CONFUSION MATRIX

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

#CLASSIFICATION REPORT

from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred))


# TEST ON CUSTOM DATA

sample = [[
    1,      # PassengerId
    3,      # Pclass
    0,      # Sex (male)
    22,     # Age
    1,      # SibSp
    0,      # Parch
    7.25,   # Fare
    0,      # Embarked_C
    0,      # Embarked_Q
    1       # Embarked_S
]]

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("Custom Passenger Prediction:", prediction)

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")
