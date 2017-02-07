import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import random
import csv
import pdb
import json

from pyspark import SparkContext
from recsys.evaluation.prediction import MAE

import csv, io

def list_to_csv(x):
	output = io.StringIO("")
	csv.writer(output).writerow(x)
	return output.getvalue().strip()

def parseV(line):
	key = line['reviewerID']
	value1 = line['asin']
	value2 = line['overall']
	return key,(value1,float(value2))


'''
    For users with # interactions > n, replace their interaction history
    with a sample of n products_with_rating
'''
def sampleInteractions(user_id,products_with_rating,n):
    if len(products_with_rating) > n:
        return user_id, random.sample(products_with_rating,n)
    else:
        return user_id, products_with_rating


'''
    For each user, find all product-product pairs combos. (i.e. products with the same user) 
'''
def findproductPairs(user_id,products_with_rating):
    for product1,product2 in combinations(products_with_rating,2):
        return (product1[0],product2[0]),(product1[1],product2[1])

def calcSim(product_pair,rating_pairs):
	''' 
    For each product-product pair, return the specified similarity measure,
    along with co_raters_count
	'''
	sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
	for rating_pair in rating_pairs:
		sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
		sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
		sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
		# sum_y += rt[1]
		# sum_x += rt[0]
		n += 1
    
	cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
	return product_pair, (cos_sim,n)

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared
    return (numerator / (float(denominator))) if denominator else 0.0

    '''
    For each product-product pair, make the first product's id the key
    '''

def keyOnFirstproduct(product_pair,product_sim_data):
    
    (product1_id,product2_id) = product_pair
    return product1_id,(product2_id,product_sim_data)

'''
    Sort the predictions list by similarity and select the top-N neighbors
'''

def nearestNeighbors(product_id,products_and_sims,n):
    
	list(products_and_sims).sort(key=lambda x: x[1][0],reverse=True)
	return product_id, list(products_and_sims)[:n]


def topNRecommendations(user_id,products_with_rating,product_sims,n):
    '''
    Calculate the top-N product recommendations for each user using the 
    weighted sums method
    '''

    # initialize dicts to store the score of each individual product,
    # since an product can exist in more than one product neighborhood
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for (product,rating) in products_with_rating:

        # lookup the nearest neighbors for this product
        nearest_neighbors = product_sims.get(product,None)

        if nearest_neighbors:
        	for (neighbor,(sim,count)) in nearest_neighbors:
        		if neighbor != product:

        		# update totals and sim_sums with the rating data
        			totals[neighbor] += sim * rating
        			sim_sums[neighbor] += sim

    # create the normalized list of scored products 
    scored_products = [(total/sim_sums[product],product) for product,total in totals.items()]
    # sort the scored products in ascending order
    scored_products.sort(reverse=True)

    # take out the product score
    # ranked_products = [x[1] for x in scored_products]
    return user_id,scored_products[:n]







if __name__ == "__main__":
	if len(sys.argv) != 2:
        	print('Usage: <input file> <output file>')
        	exit(-1)

	sc = SparkContext()

	lines = sc.textFile(sys.argv[1],1)
	'''
	user_id -> [(product_id_1, rating_1),
                (product_id_2, rating_2),
                ...]
	'''

	user_product_pairs = lines.map(json.loads).map(parseV).groupByKey().map(lambda p: sampleInteractions(p[0],p[1],500)).cache()

	# Split the data for training and testing 
    split = user_product_pairs.randomSplit((0.25,0.75))
	training = split[1].cache()
	test = split[0].cache()
    	test = test.collect()
	'''
	(product1,product2) ->    [(product1_rating,product2_rating),
                         (product1_rating,product2_rating),
                             ...]
	'''
	pairwise_products = training.filter(lambda p: len(p[1]) > 1).map(lambda p: findproductPairs(p[0],p[1])).groupByKey()

	'''
    	Calculate the cosine similarity for each product pair and select the top-N nearest neighbors:
        	(product1,product2) ->    (similarity,co_raters_count)
	'''

	product_sims1 = pairwise_products.map(
		lambda p: calcSim(p[0],p[1])).map(
		lambda p: keyOnFirstproduct(p[0],p[1])).groupByKey()
	product_sims = product_sims1.map(lambda p: nearestNeighbors(p[0],p[1],50)).collect()


	'''
    Preprocess the product similarity matrix into a dictionary and store it as a broadcast variable:
	'''

	product_sim_dict = {}
	for (product,data) in product_sims: 
		product_sim_dict[product] = data

	isb = sc.broadcast(product_sim_dict)

	'''
    Calculate the top-N product recommendations for each user
        user_id -> [product1,product2,product3,...]
	'''
	user_product_recs = training.map(
		lambda p: topNRecommendations(p[0],p[1],isb.value,50))

	#for(key, value) in user_product_recs.collect():
	#	print('user/product'+key+'has following recommendation'+'[%s]' % ', '.join(map(str, value)))
	
	#op = user_product_recs.map(list_to_csv)
	#user_product_recs.saveAsTextFile(sys.argv[2])

    user_product_recs = user_product_recs.collect()
    	preds = []
    	for (user,items_with_rating) in user_product_recs:
        	for (rating,item) in items_with_rating:
            		for (user_t,test_rating) in test:
                		if str(user_t) == str(user):
                    			for (item_t,rating_t) in test_rating:
                        			if str(item) == str(item_t):
                            				preds.append((rating,float(rating_t)))


    	mae = MAE(preds)
    	result = mae.compute()
    	print("Mean Absolute Error: ",result)