import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'Titanic-Dataset.csv'
data = pd.read_csv(file_path)

column_to_exclude = 'Survived'

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

test_data_without_column = test_data.drop(columns=[column_to_exclude])

train_data.to_csv('Titanic_Train.csv', index=False)
test_data_without_column.to_csv('Titanic_Test.csv', index=False)

print("Train and test datasets have been created and saved as 'Titanic_Train.csv' and 'Titanic_Test.csv'.")