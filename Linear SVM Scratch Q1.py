# importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import LinearSVC
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools as it

################################################################### Q 1 - 1 ##############################################################
# loading data of two datasets
dataset1 = pd.read_csv('hw3_dataset1.csv',names=['feature1','feature2','class']) # importing the first dataset
dataset2 = pd.read_csv('hw3_dataset2.csv',names=['feature1','feature2','class']) # importing the second dataset
#print("first dataset: \n",dataset1)
#print("second dataset: \n",dataset2)

# visualizing the dataset 1 and dataset 2 using the scatter plot
b = dataset1['class'] == 0
g = dataset1['class'] == 1
r = dataset2['class'] == 0
m = dataset2['class'] == 1
#print(dataset1['feature1'][b])

# visulazing the dataset-1
plt.scatter(dataset1['feature1'][b],dataset1['feature2'][b],c='b',label='class 0')
plt.scatter(dataset1['feature1'][g],dataset1['feature2'][g],c='g',label='class 1')
plt.legend()
plt.title("Dataset 1 visualization")
plt.show()

# visulazing the dataset-2
plt.scatter(dataset2['feature1'][r],dataset2['feature2'][r],c='r',label='class 0')
plt.scatter(dataset2['feature1'][m],dataset2['feature2'][m],c='m',label='class 1')
plt.legend()
plt.title("Dataset 2 visualization")
plt.show()

################################################################### Q 1 - 2 and 3 ##############################################################

# a function to plot the subplot of the given dataset
def draw_svm(dataset):
    # splitting the dataset into X and y 
    X = dataset[['feature1','feature2']]
    y = dataset['class']
    # converting the datasets into the arrays
    X = np.array(X)
    y = np.array(y)
    # different values of C to work on
    C = [0.001, 0.01, 0.1, 1]
    # gridspace for the subplot pltting
    gs = gridspec.GridSpec(2, 2)
    # figure to plot the subplots
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
    labels = ['C = 0.001','C = 0.01','C = 0.1','C = 1']

    # for loop for 4 subplot 
    for i,grd,t in zip(C,it.product([0, 1], repeat=2),labels):

        # Plotting the Points
        ax = plt.subplot(gs[grd[0], grd[1]])
        ax.scatter(X[:,0], X[:,1], c=y)
        # The SVM Model with given C parameter
        clf = LinearSVC(C=i)
        clf_fit = clf.fit(X, y)
        print("The accuracy of C: ",i," is equal to ",clf.score(X,y))
        # getting the min and max limit of X and y
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
        # Creating the meshgrid
        xx = np.linspace(x_min, x_max, 30)
        yy = np.linspace(y_min,y_max, 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf_fit.decision_function(xy).reshape(XX.shape)
    
        # Plotting the margin
        ax.contour(XX, YY, Z, colors='k', levels=[-0.05, 0, 0.05], 
                        alpha=0.5, linestyles=['--', '-', '--'])
        # plotting the decision region
        plot_decision_regions(X, y, clf=clf_fit,legend=2,ax=ax)
        # giving title to each subplot
        ax.set_title(t)
    plt.show()

# plotting subplots on the first dataset with different value of C and linearSVM
print("Plotting the decision boundaries for dataset1")
draw_svm(dataset1)
# plotting subplots on the second dataset with different value of C and linearSVM
print("Plotting the decision boundary for the dataset2")
draw_svm(dataset2)