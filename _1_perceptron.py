import numpy
import pandas
import pickle

class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = numpy.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                print(xi)
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        print(self.w_[1:], self.w_[0])
        return numpy.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return numpy.where(self.net_input(X) >= 0, 1, -1) > 0


dataframe=pandas.read_csv('iris.csv')

dataframe = dataframe[(dataframe['variety'] != 'Virginica')]

x=dataframe.iloc[:,0:4].to_numpy()
y=numpy.where(dataframe['variety']=='Setosa',-1,1)

model=Perceptron()
model.fit(x,y)
print("Koniec")
file_name=open('model_nauczony.pkl','wb')
pickle.dump(model,file_name)
