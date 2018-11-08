"""
@author: Matheus Santos Araujo
"""
import pandas as pd
from sklearn.datasets import load_digits

# Location of dataset
url = "phishing.csv"

# Assign colum names to the dataset
names = ['SFH', 'popUpWidnow', 'SSLfinal_State', 'Request_URL', 'URL_of_Anchor', 'web_traffic', 'URL_Length', 'age_of_domain', 'having_IP_Address', 'Result']

# Read dataset to pandas dataframe
data = pd.read_csv(url, names=names)  

X = data[['SFH', 'popUpWidnow', 'SSLfinal_State', 'Request_URL', 'URL_of_Anchor', 'web_traffic', 'URL_Length', 'age_of_domain', 'having_IP_Address']]
y = data['Result']

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=1000, activation='logistic', tol=0.0001, solver = 'lbfgs')  
mlp.fit(X_train, y_train.values.ravel())  

predictions = mlp.predict(X_test)  

digits = load_digits()
X, y = digits.data, digits.target

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions)) 
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f\n" % mlp.score(X_test, y_test))
