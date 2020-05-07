import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
data = pd.read_csv("processed.csv")
data = np.array(data)
X = data[:, 1:11]
y = data[:, 0]
y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LinearRegression()
log_reg.fit(X_train, y_train)
inputt=[int(x) for x in "94 94 32 92 92 96 39 66 3 5".split(' ')]
final=[np.array(inputt)]
b = log_reg.predict(final)

pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))