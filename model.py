import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = pd.concat([happy,angry,sad,fair,neutral,suprised], ignore_index=True)
y = [1]*len(happy) + [2]*len(angry) + [3]*len(sad) + [4]*len(fair) + [5]*len(neutral) + [6]*len(suprised)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

svm_model = svm.SVC(kernel='poly', C=1)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)