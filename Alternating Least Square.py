import sys
from pyspark import SparkContext
import numpy as np
from numpy.random import rand
from numpy import matrix

        
# calulate RMSE        
def rmse(user_rating, products, users):
    prod_usr = products * users.T
    diff = user_rating - prod_usr
    sqDiff = (np.power(diff, 2)) / (products_row * users_row)
    return np.sqrt(np.sum(sqDiff))
    
def mapUsr(x):  
    user = users_brod.value.T * users_brod.value
    for a in range(feature):
        user[a, a] = user[a,a] + lam * n

    new_user = users_brod.value.T * user_rating_brod.value[x,:].T
    return np.linalg.solve(user, new_user)

def mapProd(x):
    ur = user_rating_brod.value.T
    product = products_brod.value.T * products_brod.value
    for b in range(feature):
        product[b, b] = product[b,b] + lam * m
        
    new_product = products_brod.value.T * ur[x, :].T
    return np.linalg.solve(product, new_product)
   
if __name__ == "__main__":

    sc = SparkContext()
    
# Initializing parameters
    lam = 0.001
    iteration =  10
    i = 0
    feature = 10
    rms = np.zeros(iteration)
   
# Loading amazon products data in to the matrix
    lines = sc.textFile(sys.argv[1])
    parts = lines.map(lambda l: l.split(","))
    
    user_rating = np.matrix(parts.collect()).astype('float')
    m,n = user_rating.shape
    
    user_rating_brod = sc.broadcast(user_rating)

# create weight matrix using rating matrix
    w = np.zeros(shape=(m,n))
    for r in range(m):
        for j in range(n):
            if user_rating[r,j]>0.5:
                w[r,j] = 1.0
            else:
                w[r,j] = 0.0
      
# Randomly create products and users matrix
    products = matrix(rand(m, feature))
    products_brod = sc.broadcast(products)
    
    users = matrix(rand(n, feature))
    users_brod = sc.broadcast(users)
    
    products_row,products_col = products.shape
    users_row,users_col = users.shape

# iterate until products and users matrix converge
    while (i<iteration):
      
# solving products matrix by keeping user matrix constant
        products = sc.parallelize(range(products_row)).map(mapUsr).collect()
        products_brod = sc.broadcast(matrix(np.array(products)[:, :]))

# solving user matrix by keeping product matrix constant 
        users = sc.parallelize(range(users_row)).map(mapProd).collect()
        users_brod = sc.broadcast(matrix(np.array(users)[:, :]))

        error = rmse(user_rating, matrix(np.array(products)), matrix(np.array(users)))
        rms[i] = error
        i = i+1
    
    sqUser = np.array(users).squeeze()
    sqProduct = np.array(products).squeeze()
    final = np.dot(sqProduct,sqUser.T)
    
# removing already rated products to get only recommendations
    recProd = np.argmax(final - 5 * w,axis =1)
    
# printing predicted products
    for u in range(products_row):
        r = recProd.item(u)
        p = final.item(u,r)
        print ('Prediction : User %d has recommended product # %d, with rating %d' %(u+1,r+1,p) )
        
    print "RMSE after each iteration : ",rms
    
    print "Avg RMSE : ",np.mean(rms)
    sc.stop()

  

