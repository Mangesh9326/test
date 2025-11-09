import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('dataml/SAheart.csv')  
df['famhist'] = df['famhist'].map({'Present': 1, 'Absent': 0})

X = df.drop('chd', axis=1)  
y = df['chd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   

scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)  
model.fit(X_train_scaled, y_train)  

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)  
print("Test accuracy: {:.2f}%".format(acc*100))  
print(classification_report(y_test, y_pred)) 
