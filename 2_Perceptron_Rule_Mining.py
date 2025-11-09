import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron  
from sklearn.metrics import accuracy_score, classification_report  
  
df = pd.read_csv('dataml/SAHeart.csv')   
print(df.columns)
# Convert 'famhist' from strings to numbers  
df['famhist'] = df['famhist'].map({'Present': 1, 'Absent': 0})  
  
# Choose features and label 
X = df.drop('chd', axis=1)
y = df['chd']  
  
# Split into train/test  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
  
# Scale features  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  
  
perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=42) 
perceptron.fit(X_train_scaled, y_train)  
  
y_pred = perceptron.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy (Perceptron): {:.2f}%".format(accuracy * 100))  
print(classification_report(y_test, y_pred))  
