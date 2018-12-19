import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

#***** *****examples

#*****Пример с линейной регрессией
#from sklearn.linear_model import LinearRegression

#rng = np.random.RandomState(42)
#x = 10 * rng.rand(50)
#y = 2 * x - 1 + rng.randn(50)

#model = LinearRegression(fit_intercept=True)
#X = x[:, np.newaxis] 

#model.fit(X, y)
#print("model angular coefficient:")
#print(model.coef_)
#print("model point of intersection:")
#print(model.intercept_)

#xfit = np.linspace(-1, 11)
#Xfit = xfit[:, np.newaxis]
#yfit = model.predict(Xfit)

#plt.scatter(x, y)
#plt.plot(xfit, yfit) 
#plt.show()

#*****подготовка данных
#irisData = pd.read_csv('iris.csv')
#X_iris = irisData.iloc[0:150, [0,2]].values
#y_iris = irisData.iloc[0:150, [4]].values.flatten()

#print(X_iris)
#print(y_iris)

#*****Классификация набора данных
#from sklearn.cross_validation import train_test_split
#from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score

#Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)

#model = GaussianNB() 
#model.fit(Xtrain, ytrain) 
#y_model = model.predict(Xtest)

#score = accuracy_score(ytest, y_model) 

#print(score)

#*****Обучение без учителя
#from sklearn.decomposition import PCA
#import seaborn as sns

#model = PCA(n_components=2) 
#model.fit(X_iris)  
#X_2D = model.transform(X_iris) 
#irisData['PCA1'] = X_2D[:, 0]
#irisData['PCA2'] = X_2D[:, 1]
#sns_plot = sns.lmplot("PCA1", "PCA2", data=irisData, fit_reg=False)
#sns_plot.savefig("sns_plot.png")

##Кластеризацию набора данных 
#from sklearn.mixture import GMM

#model = GMM(n_components=3, covariance_type='full') 
#model.fit(X_iris)   
#y_gmm = model.predict(X_iris)

#irisData['cluster'] = y_gmm
#sns_plot = sns.lmplot("PCA1", "PCA2", data=irisData, col='cluster', fit_reg=False)
#sns_plot.savefig("sns_plot1.png")

#*****Анализ рукописных цифр 
#from sklearn.datasets import load_digits
#digits = load_digits() 
#digits.images.shape

#fig, axes = plt.subplots(10, 10, figsize=(8, 8),subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
#for i, ax in enumerate(axes.flat):
#  ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
#  ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green') 

#plt.show()


#X = digits.data
#y = digits.target

#from sklearn.manifold import Isomap
#iso = Isomap(n_components=2)
#iso.fit(digits.data)
#data_projected = iso.transform(digits.data)
#data_projected.shape 

#plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('Blues', 10))
#plt.colorbar(label='digit label', ticks=range(10))
#plt.clim(-0.5, 9.5);
#plt.show()

#*****Классификация цифр
#from sklearn.cross_validation import train_test_split
#Xtrain, Xtest, ytrain, ytest =  train_test_split(X, y, random_state=0)
#from sklearn.naive_bayes import GaussianNB
#model = GaussianNB()
#model.fit(Xtrain, ytrain)
#y_model = model.predict(Xtest) 

#from sklearn.metrics import accuracy_score

#print(accuracy_score(ytest, y_model))

#from sklearn.metrics import confusion_matrix
#import seaborn as sns
#mat = confusion_matrix(ytest, y_model)
#sns.heatmap(mat, square=True, annot=True, cbar=False)
#plt.xlabel('predicted value')
#plt.ylabel('true value');
 

#fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
#for i, ax in enumerate(axes.flat):
#  ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
#  ax.text(0.05, 0.05, str(y_model[i]), transform=ax.transAxes, color='green' if (ytest[i] == y_model[i]) else 'red')
  
#plt.show()

#***** *****end of examples
















titanicData = pd.read_csv('train.csv')

filteredData = titanicData.query('Age == Age')
#print(filteredData)

X_titanic = filteredData[["Pclass", "Age"]].values
y_titanic = filteredData[["Survived"]].values.flatten()

print(X_titanic)
print(y_titanic)

#X_titanic[X_titanic == "male"] = 1
#X_titanic[X_titanic == "female"] = 2
#print(X_titanic)

from sklearn.decomposition import PCA
import seaborn as sns

model = PCA(n_components=2) 
model.fit(X_titanic)  
X_2D = model.transform(X_titanic) 
print(X_2D)

filteredData['PCA1'] = X_2D[:, 0]
filteredData['PCA2'] = X_2D[:, 1]

sns_plot = sns.lmplot("PCA1", "PCA2", data=filteredData, fit_reg=False)
sns_plot.savefig("sns_titanic.png")

from sklearn.mixture import GMM

model = GMM(n_components=3, covariance_type='full') 
model.fit(X_titanic)   
y_gmm = model.predict(X_titanic)

filteredData['cluster'] = y_gmm
sns_plot = sns.lmplot("PCA1", "PCA2", data=filteredData, col='cluster', fit_reg=False)
sns_plot.savefig("sns_titanic1.png")