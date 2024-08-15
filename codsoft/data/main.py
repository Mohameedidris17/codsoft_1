import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


train_data = pd.read_csv('Titanic_Train.csv')
test_data = pd.read_csv('Titanic_Test.csv')

train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy}')

test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'].fillna(test_data['Fare'].median())

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)

test_predictions = model.predict(test_data)

submission = pd.DataFrame({
    'PassengerId': pd.read_csv('Titanic_Test.csv')['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('titanic_submission.csv', index=False)
print("titanic_submission.csv are successfully created")