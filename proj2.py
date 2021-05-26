# -*- coding: utf-8 -*-
######################################################################
#EEE 591 : Python for Rapid Engineering Solutions, Fall 2020         #
#Project 2                                                           #
#Classifying Rocks and Mines from Sonar Data using PCA               #
#Author: Mujahed Syed                                                #
#Instructor: Dr. Steven Millman                                      #
######################################################################

#Importing the Required Packages

import pandas as pd                                            #For reading the dataset
import numpy as np                                             #For creating the x-axis of the plot
from warnings import filterwarnings                            #For supressing warnings 
from sklearn.model_selection import train_test_split           #For Creating a train test split of the dataset       
from sklearn.preprocessing import StandardScaler               #For Standard scaling the features
from sklearn.decomposition import PCA                          #For performing PCA on features (Dimensionality Reduction)
from sklearn.metrics import accuracy_score,confusion_matrix    #For evaluating the performance of the Algorithm
from sklearn.neural_network import MLPClassifier               #Multi-Layered Perceptron
import matplotlib.pyplot as plt                                #For creating the plot

np.random.seed(42)                                             #Setting the random seed to reproduce the same results

filterwarnings('ignore')                                       #Ignoring the warnings

df = pd.read_csv("sonar_all_data_2.csv", header=None)          #Reading in the dataset and specifying that the data does not have a header

features = df.iloc[:,:-1].values                               #Separating the features
labels = df.iloc[:,-1].values                                  #Separating the labels 

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.3, random_state=0)    #Performing the train test split

sc = StandardScaler()                                         #Creating an instance of the Standard Scaler

sc.fit(X_train)                                               #Fitting the training data(features) to the instance of the standard scaler
X_train_std = sc.transform(X_train)                           #Transforming the training data (features)
X_test_std = sc.transform(X_test)                             #Transforming the test data (features)

accuracy = []                                                 #Initializing a list to store accuracies
conf_mats = []                                                #Initializing a list to store confusion matrices

for n_comps in range(1,61):                                   #Iterating through 1 to 60 and accordingly updating the no of components
    pca_n = PCA(n_components = n_comps)                       #Creating an instance of PCA and specifying how many components we need
    X_train_pca = pca_n.fit_transform(X_train_std)            #Fitting the training features to the pca instance and transforming them 
    X_test_pca = pca_n.transform(X_test_std)                  #Transforming the test data features
    #Defining the MLP classifier and specifying the parameters
    mlp_model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', max_iter=2000, alpha=0.00001, solver='lbfgs', tol=0.0001)
    mlp_model.fit(X_train_pca,Y_train)                         #Fitting the pca transformed training features and the training labels
    Y_pred = mlp_model.predict(X_test_pca)                     #Performing the predictions on the test data
    acc_n = accuracy_score(Y_test, Y_pred)                     #Calculating the test accuracy score
    accuracy.append(acc_n)                                     #Appending the accuracy score to the accuracy list
    print("Accuracy for",n_comps,"components is:",acc_n)       #Printing to the console, the test accuracy for each iteration 
    conf_mat = confusion_matrix(Y_test, Y_pred)                #Generating the confusion matrix for each iteration
    conf_mats.append(conf_mat)                                 #Appending the confusion matrix generated in previous step to the list of confusion matrices   

best_acc = max(accuracy)                                                       #Getting the best accuracy score from the list of accuracies
best_comps = accuracy.index(best_acc)                                          #Getting the number of components returned by pca for the best accuracy

print(" ")                                                                         #For better readability on the console
print("We get the best accuracy of:",best_acc,"for",best_comps+1,"components.")    #Printing the best accuracy and the corresponding number of components 

#Confusion Matrix of Best Accuracy
print(" ")                                                                     #For better readability
print("The confusion matrix when we use",best_comps+1,"components:")           
print(" ")                                                                     #For better readability
print(conf_mats[best_comps+1])                                                 #Printing the confusion matrix for the best accuracy
    
#Plotting
plt.plot(np.arange(1, 61), accuracy)                                           #Plotting number of components vs accuracy
plt.xlabel('Number of Components')                                             #Setting the x-axis label
plt.ylabel('Accuracy')                                                         #Setting the y-axis label 
plt.title('Number of Components vs Accuracy')                                  #Giving the plot a title
plt.show()                                                                     #Displaying the plot
    