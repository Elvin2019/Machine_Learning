from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('turboaz.csv') #first loading data


#Loading data (Grade: 10%)
X1 = dataset['Yurush'].map(lambda x: x.rstrip("km").replace(' ','')).map(int)
X2 = dataset['Buraxilish ili'] #Buraxilish ili
y = dataset['Qiymet'].map(lambda x: float(x.rstrip('$'))*1.7 if '$' in x else float(x.rstrip('AZN')))  #Qiymet

#VISUALIZATION PART-1 ALL 3 REQUIREMENTS WAS DONE (GRADE: 10%)

plt.figure(1)
plt.scatter(X1, y, color ='b')
plt.xlabel('X Label - Yurush')
plt.ylabel('Y Label - Qiymet')


plt.figure(2)
plt.scatter(X2, y, color = 'r')
plt.xlabel('X Label - Buraxilish ili')
plt.ylabel('Y Label - Qiymet')


fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d') #3d projection
ax.scatter(X1, X2, y, c = 'g') 
ax.set_xlabel('X Label - Yurush')
ax.set_ylabel('Y Label - Buraxilish ili')
ax.set_zlabel('Z Label - Qiymet')


#save featres and result for future 
X1_s = X1
X2_s = X2
y_s = y
#normalize data 
X1 = (X1 - X1.mean()) / X1.std()
X2 = (X2 - X2.mean()) / X2.std()
y =  (y - y.mean()) / y.std() 


m = len(X1)
ones = np.ones(m)
X = np.array([ones, X1, X2]).T
theta = np.array([0,0,0])


#Cost function (GRADE: 20%)
def computeCost(X, y, theta):
    m = len(y)
    h_x = X.dot(theta) #compute h_theta(x) = theta^T * x
    J = np.sum((h_x - y)**2)/2/m
    return J

J = computeCost(X, y, theta)
print(J)




#Gradient descent from scratch (GRADE: 40%)
iterations = 10000
alpha = 0.001 #learning rate


def gradientDescent(X, y, theta, alpha, iterations):
    J_history = [0] * iterations
    m = len(y)
    
    for i in range(iterations):
        if i % 1000 == 0:
            print("iteration #%d" % i)
            print(computeCost(X,y,theta))    
        #hypothesis    
        h_x = X.dot(theta)
        theta = theta - alpha/m * (X.T.dot(h_x-y)) 
    
        cost = computeCost(X,y,theta)
        J_history[i] = cost  #Save the cost J in every iteration 
    return theta, J_history

new_theta, J_history = gradientDescent(X, y, theta, alpha, iterations)


#VISUALIZATION PART-2 ALL 4 REQUIREMENTS WAS DONE

#visualization for  array of cost at each iteratin 
plt.plot(J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost_error')
plt.title("Array of costs at each iteration ")


#visualization for line of predictions that made with parameters wia gradientDescent 
#first plot for Yurush and Qiymet
plt.figure(4)
plt.scatter(X1, y, color ='b')
plt.xlabel('X Label - Yurush')
plt.ylabel('Y Label - Qiymet')
prediction = new_theta[1] * X1 + new_theta[0]
plt.plot(X1, prediction, c ='r')
#plt.scatter(X1, prediction, c = 'r' )

#Second plot for Buraxilish ili and Qiymet

plt.figure(5)
plt.scatter(X2, y, color = 'b')
plt.xlabel('X Label - Buraxilish ili')
plt.ylabel('Y Label - Qiymet')
prediction = new_theta[2] * X2 + new_theta[0]
plt.plot(X2, prediction, c ='r')
#plt.scatter(X2, prediction, c = 'r' )

fig = plt.figure(6)
ax = fig.add_subplot( projection='3d') #3d projection
ax = Axes3D(fig)
ax.scatter(X1, X2, y, c = 'b') 
ax.set_xlabel('X Label - Yurush')
ax.set_ylabel('Y Label - Buraxilish ili')
ax.set_zlabel('Z Label - Qiymet')
prediction = new_theta[1] * X1 + new_theta[2] * X2 + new_theta[0]
ax.scatter(X1, X2, prediction, c = 'r' )






###############################################################################
#TESTING PART (GRADE: 20%)
# first car proporties
yurush1 = 240000
b_ili1 = 2000
qiymet1 = 11500

#now we will use feature scaling method to test this data
#since we have already normalise our data before therefore
#we need also to normalise this testing data

yurush1_S = (yurush1 - X1_s.mean()) / X1_s.std()
b_ili1_S = (b_ili1 -X2_s.mean()) / X2_s.std()
qiymet1_S = (qiymet1 - y_s.mean()) / y_s.std()

pre_qiymet_1 = new_theta[1] * yurush1_S + new_theta[2] * b_ili1_S + new_theta[0]


#now we need to go backward from normalize mode then we can see real price
pre_qiymet_1 = pre_qiymet_1 * y_s.std() + y_s.mean()
real_qiymet_1 = qiymet1_S * y_s.std() + y_s.mean()

print('Real qiymet: ', real_qiymet_1,'   ', 'prediction: ',pre_qiymet_1)



# second car
yurush2 = 415558
b_ili2 = 1996
qiymet2 = 8800

yurush2_S = (yurush2 - X1_s.mean()) / X1_s.std()
b_ili2_S = (b_ili2 -X2_s.mean()) / X2_s.std()
qiymet2_S = (qiymet2 - y_s.mean()) / y_s.std()

pre_qiymet_2 = new_theta[1] * yurush2_S + new_theta[2] * b_ili2_S + new_theta[0]


#now we need to go backward from normalize mode then we can see real price
pre_qiymet_2 = pre_qiymet_2 * y_s.std() + y_s.mean()
real_qiymet_2 = qiymet2_S * y_s.std() + y_s.mean()

print('Real qiymet: ',  real_qiymet_2, '   ', 'prediction: ',  pre_qiymet_2)


###############################################################################





#Linear Regression using library( 20% OF GRADE)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

X_train = []

 #array of two features including values
for j in range(m):
    X_train.append([X1_s[j], X2_s[j]])
        
        
        
        
X_test = [[240000, 2000], [415558, 1996]] #X_test for yurush ve buraxilish ili 
y_test = [11500, 8800] #y_tes for qiymet

regression = linear_model.LinearRegression()
regression.fit(X_train, y_s) #fiting data to trin for finding cooeficents

y_pre = regression.predict(X_test)
print('Real qiymet: ', y_test,' ', 'preidction: ', y_pre) 

#several proporties of this library that makes it efficient to find coefficiants, mean_squared error

print("Coefficents:", regression.coef_) #coefficients
print("M.S.E:", mean_squared_error(y_test, y_pre)) #mean squared error
print("Variace", r2_score(y_test, y_pre))

#Visulaization 

plt.figure(7)
plt.scatter(X1_s, y_s, color ='b')
plt.xlabel('X Label - Yurush')
plt.ylabel('Y Label - Qiymet')
prediction = regression.coef_[0] * X1_s + 15000
plt.plot(X1_s, prediction, c ='r')

plt.figure(8)
plt.scatter(X2_s, y_s, color = 'b')
plt.xlabel('X Label - Buraxilish ili')
plt.ylabel('Y Label - Qiymet')
prediction = regression.coef_[1] * X2_s +regression.intercept_
plt.plot(X2_s, prediction, c ='r')

fig = plt.figure(9)
ax = fig.add_subplot( projection='3d') #3d projection
ax = Axes3D(fig)
ax.scatter(X1_s, X2_s, y_s, c = 'b') 
ax.set_xlabel('X Label - Yurush')
ax.set_ylabel('Y Label - Buraxilish ili')
ax.set_zlabel('Z Label - Qiymet')
predict = regression.coef_[0] * X1_s + regression.coef_[1] * X2_s + regression.intercept_
ax.scatter(X1_s, X2_s, predict, c = 'r' )

###############################################################################
#EXTRA TASK (GRADE: 40%)
#. Solve linear regression by Normal equation(20 %) 

from numpy.linalg import inv

def normal_equation(X, y):  
    theta = inv(X.T.dot(X)).dot(X.T).dot(y)  
    # normal equation  
    # theta = (X.T * X)^(-1) * X.T * y  
      
    return theta # returns a list  

Y = (y[:, np.newaxis])
N_theta = normal_equation(X,Y) # finding theta

#visualisation
#first plot for Yurush and Qiymet

plt.figure(10)
plt.scatter(X1, y, color ='b')
plt.xlabel('X Label - Yurush')
plt.ylabel('Y Label - Qiymet')
pr_ = N_theta[1] * X1 + N_theta[0]
plt.plot(X1, pr_, c ='r')

#Second plot for Buraxilish ili and Qiymet

plt.figure(11)
plt.scatter(X2, y, color = 'b')
plt.xlabel('X Label - Buraxilish ili')
plt.ylabel('Y Label - Qiymet')
pr_ = N_theta[2] * X2 + N_theta[0]
plt.plot(X2, pr_, c ='r')

fig = plt.figure(12)
ax = fig.add_subplot( projection='3d') #3d projection
ax = Axes3D(fig)
ax.scatter(X1, X2, y, c = 'b') 
ax.set_xlabel('X Label - Yurush')
ax.set_ylabel('Y Label - Buraxilish ili')
ax.set_zlabel('Z Label - Qiymet')
pr_ = N_theta[1] * X1 + N_theta[2] * X2 + N_theta[0]
ax.scatter(X1, X2, pr_, c = 'r' )

###############################################################################

#Instead of linear function use polynomial which over
#performed linear hypothesis accuracy (20%)
#we can define polynomial function with scratch or we can use sklearn PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression

X_t, y_t = make_regression(n_samples = 300, n_features=1, noise=8, bias=2)
y2 = y_t**2

poly_features = PolynomialFeatures(degree = 5)  
X_poly = poly_features.fit_transform(X_t)
poly_model = LinearRegression()  
poly_model.fit(X_poly, y2)

pred = poly_model.predict(X_poly)
new_X, new_y = zip(*sorted(zip(X_t, pred))) # sort values for plotting
plt.plot(new_X, new_y, c = 'r')
plt.scatter(X_t,y2, c = 'b')


#if we increased the degree of thetat in this case accuracy will be mor accurate

