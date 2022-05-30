from flask import Flask, render_template, request
import pickle
import numpy

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
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        print(self.w_[1:], self.w_[0])
        return numpy.dot(X, self.w_[1:]) + self.w_[0]

    # Function change
    def predict(self, X):
        if numpy.where(self.net_input(X) >= 0, 1, -1) > 0:
            return "Versicolor"
        else:
            return "Setosa"


pikl=open("model_nauczony.pkl",'rb')

model=pickle.load(pikl)


app = Flask(__name__)

@app.route('/')
def index():  # put application's code here
    return render_template('index.html')

@app.route('/solve')
def solution():  # put application's code here
    return model.predict([
        float(request.args.get('sepal-length')),
        float(request.args.get('sepal-width')),
        float(request.args.get('petal-length')),
        float(request.args.get('petal-width'))
    ])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
