#import libraries----------------------------------------------
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


import matplotlib.pyplot as plt


import graphviz
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image  
import pydotplus

import keras.models
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

#Tree 1
col_names = ['Food', 'Electricity', 'Apparel', 'New Vehicles','Target']
pima = pd.read_csv("CPI_first.csv", header=None, names=col_names)

feature_cols = ['Electricity','Food', 'Apparel', 'New Vehicles']
X = pima[feature_cols]
y = pima.Target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


#Decision Tree for Tree 1----------------------------------------------

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


#CM before pruning Tree 1----------------------------------------------

cm = confusion_matrix(y_test,y_pred)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))

#Write Tree 1 to file----------------------------------------------
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Inflation11.png')
Image(graph.create_png())
graph

#Prune Tree 1: set to entropy,split----------------------------------------------

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3,min_samples_split=70)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#Tree 1 CM after pruning----------------------------------------------

cm = confusion_matrix(y_test,y_pred)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))

#Write pruned Tree 1 to file

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Inflation12.png')
Image(graph.create_png())






#Tree 2
col_names = ['Energy', 'Gas', 'EducationAndCommunication', 'MedicalServices','Target']
pima = pd.read_csv("CPI_second.csv", header=None, names=col_names)

feature_cols = ['Energy', 'Gas', 'EducationAndCommunication', 'MedicalServices']
X = pima[feature_cols]
y = pima.Target 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


#Tree 2 Decision Tree----------------------------------------------

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


#Tree 2 CM before pruning----------------------------------------------

cm = confusion_matrix(y_test,y_pred)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))

#Write Tree 2 to file----------------------------------------------
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Inflation21.png')
Image(graph.create_png())
graph

#Prune Tree 2,set to entropy,split----------------------------------------------

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3,min_samples_split=50)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#Tree 2 CM after pruning----------------------------------------------

cm = confusion_matrix(y_test,y_pred)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))

#Write pruned Tree 2 to file

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Inflation22.png')
Image(graph.create_png())






#Tree 3 (Mixed)
col_names = ['Apparel', 'Gas', 'Electricity', 'MedicalServices','Target']
pima = pd.read_csv("CPI_third.csv", header=None, names=col_names)

pima.head()
pima.describe()

feature_cols = ['Apparel', 'Gas', 'Electricity', 'MedicalServices']
X = pima[feature_cols]
y = pima.Target 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


#Tree 3 Decision Tree----------------------------------------------

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


#Tree 3 CM before pruning----------------------------------------------

cm = confusion_matrix(y_test,y_pred)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))

#Write Tree 3 to file----------------------------------------------
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Inflation31.png')
Image(graph.create_png())
graph

#Prune Tree 3,set to entropy,split----------------------------------------------

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3,min_samples_split=70)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#Tree 3 CM after pruning----------------------------------------------

cm = confusion_matrix(y_test,y_pred)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score:",metrics.f1_score(y_test, y_pred))

#Write pruned Tree 3 to file

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Inflation32.png')
Image(graph.create_png())




#CREATE ANN MODEL----------------------------------------------

model = Sequential()
model.add(Dense(10, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary() 

model.compile(optimizer='Adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


es = EarlyStopping(monitor='val_accuracy', 
                                   mode='max',
                                   patience=10,
                                   restore_best_weights=True)

history = model.fit(X,
                    y,
                    callbacks=[es],
                    epochs=80, 
                    batch_size=10,
                    validation_split=0.3,
                    shuffle=True,
                    verbose=1)


history_dict = history.history



loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss'] 

epochs = range(1, len(loss_values) + 1) 

#Training/Validation loss----------------------------------------------

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#Training/Validation accuracy----------------------------------------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
np.max(val_acc)

#CM for ANN----------------------------------------------

model.predict(X)
np.round(model.predict(X),0) 
y
preds = np.round(model.predict(X),0)
cm = confusion_matrix(y, preds)
cm


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

print("Accuracy:",metrics.accuracy_score(y, preds))
print("Precision:",metrics.precision_score(y, preds))
print("Recall:",metrics.recall_score(y, preds))
print("F1 score:",metrics.f1_score(y, preds))


