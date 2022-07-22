import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

def read_boston_data():
    boston = datasets.load_boston()
    df = pd.DataFrame(data = boston.data)
    df.columns = boston.feature_names
    df['Target'] = boston.target
    df= df.rename(columns = {'Target':'Price'})
    # Taking RM and LSTAT as features as they are highly correlated to price.
    x_RM = preprocessing.scale(df['RM'])
    x_LSTAT = preprocessing.scale(df['LSTAT'])
    y = preprocessing.scale(df['Price'])
    
    plt.scatter(y, x_RM, s=5, label = 'RM')
    plt.scatter(y, x_LSTAT, s=5, label = 'LSTAT')
    plt.legend(fontsize=15)
    plt.xlabel('Average number of rooms & Low status population', fontsize=15)
    plt.ylabel('Price', fontsize=15)
    plt.legend()
    plt.show()
    
    return x_RM, x_LSTAT, y

def read_data():
    X1 = []
    X2 = []
    Y = []
    for line in open('data/data_2d.csv'):
        x1,x2,y= line.split(",")
        X1.append(float(x1))
        X2.append(float(x2))
        Y.append(float(y))
        
    X1= np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y)
    
    plt.scatter(Y, X1, s=5, label = 'x1')
    plt.scatter(Y, X2, s=5, label = 'x2')
    plt.legend(fontsize=15)
    plt.xlabel('x1 and x2', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.legend()
    plt.show()
    
    x1 = preprocessing.scale(X1)
    x2 = preprocessing.scale(X2)
    y = preprocessing.scale(Y)
    
    x = np.c_[np.ones(X1.shape[0]),X1, X2]
    return x1, x2, y
    
x1,x2,y = read_data()
x = np.c_[np.ones(x1.shape[0]),x1, x2]
# Parameters required for Gradient Descent
alpha = 0.0001   #learning rate
m = y.size  #no. of samples
np.random.seed(10)
theta = np.random.rand(x.shape[1])

def gradient_descent(x, y, m, theta, alpha):
    cost_list = []   #to record all cost values to this list
    theta_list = []  #to record all theta_0 and theta_1 values to this list 
    prediction_list = []
    run = True
    cost_list.append(1e10)    #we append some large value to the cost list
    i=0
    while run:
        prediction = np.dot(x, theta)   #predicted y values theta_0*x0+theta_1*x1
        prediction_list.append(prediction)
        error = prediction - y
        cost = 1/(m) * np.dot(error.T, error) #  (1/2m)*sum[(error)^2]
        cost_list.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))   # alpha * (1/m) * sum[error*x]
        theta_list.append(theta)
        if cost_list[i]-cost_list[i+1] < 1e-9:   #checking if the change in cost function is less than 10^(-9)
            run = False

        i+=1
    cost_list.pop(0)   # Remove the large number we added in the begining 
    return prediction_list, cost_list, theta_list

prediction_list, cost_list, theta_list = gradient_descent(x, y, m, theta, alpha)
theta = theta_list[-1]

print('Model-Intercept : {}'.format(round(theta[0],3)))
print('Model-Theta_0 : {}'.format(round(theta[1],4)))
print('Model-Theta_1 : {}'.format(round(theta[2],4)))

plt.title('Cost Function J', size = 30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.plot(cost_list)
plt.show()

ys = y
xs = np.c_[x1,x2]

xs = preprocessing.scale(xs)
ys = preprocessing.scale(ys)

lm = LinearRegression()

#Fitting the model
lm = lm.fit(xs,ys)
intercept = lm.intercept_
Theta_0 = lm.coef_[0]
Theta_1 = lm.coef_[1]

print('Sklearn-Intercept : {}'.format(round(intercept,3)))
print('Sklearn-Theta_0 : {}'.format(round(Theta_0,4)))
print('Sklearn-Theta_1 : {}'.format(round(Theta_1,4)))