import numpy as np
import matplotlib.pyplot as plt
from utils import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import os

os.chdir(r"D:\repos\my_repos\lr_lz")

class One_dimensional():
    def read_and_plot(self,data):
        if data == "generate":
            self.X,self.Y = get_gen_data()
        elif data == "read":
            self.X,self.Y = get_data_1d()
        elif data == "moore":
            self.X,self.Y,self.sx, self.sy = moore_data()

        self.X1 = self.X.reshape(-1,1)
        self.Y1 = self.Y.reshape(-1,1)
        plt.scatter(self.X,self.Y,label = "Data points")

    def solve_and_fit(self,data):
        denominator = self.X.dot(self.X) - self.X.mean()**2
        m = (self.X.dot(self.Y)-self.X.mean()*self.Y.mean())/denominator
        b = (self.X.dot(self.X)*self.Y.mean()-self.X.mean()*self.X.dot(self.Y))/denominator

        self.Yhat = m*self.X+b
        plt.plot(self.X,self.Yhat,color ="red",label = "Best fit line")
        print("Slope is {} and Intercept is {}".format(np.round(m,2),np.round(b,2)))
        if data == "moore":
            print("The time taken to double the transistor count is ",np.round(np.log(2)/m*(self.sx/self.sy),2), "years.")

    def sklearn_model(self):
        model = LinearRegression()
        model.fit(self.X1,self.Y1)
        print("Sklearn model score is",model.score(self.X1,self.Y1))
        print("Sklearn model weights",model.coef_)
        print("Sklearn model Intercept", model.intercept_)

    def r_square(self):
        d1 = self.Y-self.Yhat
        d2 = self.Y-self.Y.mean()
        print("R-squared value is",1-(d1.dot(d1)/d2.dot(d2)))

class Two_dimensional():
    def read_and_plot(self):
        self.X,self.Y = get_2d_data()
        self.fig = plt.figure()
        self.ax = plt.axes(projection="3d")
        self.ax.scatter3D(self.X[:,0],self.X[:,1],self.Y)
        #plt.show()

    def solve_and_fit(self):
        self.w = np.linalg.solve(np.dot(self.X.T,self.X), np.dot(self.X.T,self.Y))
        self.Yhat = np.dot(self.X,self.w)
        #X,Y = np.meshgrid(self.X,self.Y)
        self.ax.plot_surface(self.X[:,0],self.X[:,1],self.Yhat.reshape(len(self.Yhat),1),color = 'red')
        print("Weights obtained are",self.w[1:])
        print("Bias is",self.w[0])

    def sklearn_model(self):
        model = LinearRegression()
        model.fit(self.X,self.Y)
        print("Sklearn model score is",model.score(self.X,self.Y))
        print("Sklearn model weights",model.coef_)
        print("Sklearn model Intercept", model.intercept_)


    def r_square(self):
        d1 = self.Y-self.Yhat
        d2 = self.Y-self.Y.mean()
        print("R-squared value is",1-(d1.dot(d1)/d2.dot(d2)))

class Polynomial():
    def read_and_plot(self):
        self.X,self.Y = get_poly_data()
        plt.scatter(self.X[:,1],self.Y, label = "data points")

    def solve_and_fit(self):
        self.w = np.linalg.solve(np.dot(self.X.T,self.X), np.dot(self.X.T,self.Y))
        self.Yhat = np.dot(self.X,self.w)
        plt.plot(sorted(self.X[:,1]), sorted(self.Yhat),color ="red", label = "Fit")
        print("Weights obtained are",self.w[1:])
        print("Bias is",self.w[0])

    def r_square(self):
        d1 = self.Y-self.Yhat
        d2 = self.Y-self.Y.mean()
        print("R-squared value is",1-(d1.dot(d1)/d2.dot(d2)))

    def sklearn_model(self):
        model = LinearRegression()
        model.fit(self.X,self.Y)
        print("Sklearn model score is",model.score(self.X,self.Y))
        print("Sklearn model weights",model.coef_)
        print("Sklearn model Intercept", model.intercept_)

if __name__ == '__main__':
    # which = input("Choose between\n 1. One dimensional Linear Regression\n 2. Two dimensional Linear Regression\n 3. Polynomial Regression\n ")
    which = "2"
    dict1 = {1:"lr1d",2:"lr2d",3:"poly"}
    which = dict1[int(which)]

    if which == "lr1d":
        model = One_dimensional()
        # data = input("Choose between generate, read and moore: ")
        data = 'generate'
        model.read_and_plot(data)
        model.solve_and_fit(data)


    elif which == "lr2d":
        model = Two_dimensional()
        model.read_and_plot()
        model.solve_and_fit()

    else:
        model = Polynomial()
        model.read_and_plot()
        model.solve_and_fit()

    model.r_square()
    model.sklearn_model()
    plt.legend()
    plt.show()