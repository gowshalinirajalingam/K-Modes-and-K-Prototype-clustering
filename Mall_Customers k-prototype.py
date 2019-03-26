#
#	k-prototypes.py
#
#       Run k-prototypes on (netflow) data, which inspiration (pandas) from Ed Henry
#
#	code: https://github.com/nicodv/kmodes
#
#	See "A New Approach to Data Driven Clustering", Azran, A., et al. 
#	http://mlg.eng.cam.ac.uk/zoubin/papers/AzrGhaICML06.pdf
#
#	See also
#
#	"Extensions to the k-Means Algorithm for Clustering Large Data Sets
#	with Categorical Values", Zhexue Huang
#	http://www.cse.ust.hk/~qyang/537/Papers/huang98extensions.pdf
#
#
#	David Meyer
#	dmm@1-4-5.net
#	Fri Jan 22 12:02:45 2016
#
#	$Header: $
#
#
#
#       imports
#
import sys
import time
import numpy as np
import pandas as pd
from   kmodes import kmodes
from   kmodes import kprototypes
import matplotlib.pyplot as plt

#
#
#       globals
#
DEBUG         = 2                               # set to 1 to debug, 2 for more
verbose       = 0                               # kmodes debugging. just dictates how much output gets passed to stdout (i.e. telling you what stage the algorithm is at etc).
nrows         = 201                              # number of rows to read (resources)
#
#       These are the "categorical" fields in CSV
#
categorical_field_names = ['Gender','Age']

#
#       build DataFrame
#
df = pd.DataFrame()
#
#       CSV here
#
CSV_IN  = "Mall_Customers.csv"                    # data file
CSV_OUT = "Mall_CustomersOut.csv"
#
#       read CSV into a pandas dataframe
#
#       NB: control the number of records read (nrows) here (save some resources perhaps)
#
df = pd.read_csv(CSV_IN, sep=',', nrows=nrows,header=0)
#
#       strip whitespace (should get this done at export time)
#
df.rename(columns=lambda x: x.strip(), inplace = True)
#
#       Drop NA and NaN values
#
df = df.dropna()
#
#       Ensure things are dtype="category" (cast)
#
df.dtypes
for c in categorical_field_names:
    df[c] = df[c].astype('category')
#
#       get a list of the catgorical indicies
#
categoricals_indicies = []
for col in categorical_field_names:
        categoricals_indicies.append(categorical_field_names.index(col))
#
#       add non-categorical fields
#
fields = list(categorical_field_names)
fields.append('Annual Income (k$)')
fields.append('Spending Score (1-100)')
#


#
#       select fields
#
data_cats = df.loc[:,fields]


#
#       normalize continous fields
#
#       essentially compute the z-score
#
#       note: Could use (x.max() - x.min()) instead of np.std(x)
#
columns_to_normalize     = ['Annual Income (k$)', 'Spending Score (1-100)']
data_cats[columns_to_normalize] = data_cats[columns_to_normalize].apply(lambda x: (x - x.mean()) / np.std(x))
#

data_cats.dtypes

#       kprototypes needs an array

data_cats_matrix = data_cats.as_matrix()
#
#       model parameters
#
init       = 'Huang'                  
#        {'Huang', 'Cao', 'random' or an ndarray}, default: 'Cao'
#        Method for initialization:
#        'Huang': Method in Huang [1997, 1998]
#        'Cao': Method in Cao et al. [2009]
#        'random': choose 'n_clusters' observations (rows) at random from
#        data for the initial centroids.
#        If an ndarray is passed, it should be of shape (n_clusters, n_features)
#        and gives the initial centroids

n_clusters = 4                          # The number of clusters to form as well as the number of centroids to generate.
max_iter   = 100                        # default 300.  Maximum number of iterations of the k-modes algorithm for a single run.
 
#
#       get the model
#
kproto = kprototypes.KPrototypes(n_clusters=n_clusters,init=init,verbose=verbose)
#
#       fit/predict
#
clusters = kproto.fit_predict(data_cats_matrix,categorical=categoricals_indicies)
#
#
#
#       cluster centroids

centers=kproto.cluster_centroids_
#
#
#
if (DEBUG > 2):
        print( '\nclusters:{}\nproto_cluster_assignments: {}\n'.format(clusters,zip(data_cats_matrix,clusters)))
#
#       Instantiate dataframe to house new cluster data and combine dataframe entries with resultant cluster_id
#
cluster_df = pd.DataFrame(columns=('CustomerID','Gender','Age','Annual Income (k$)', 'Spending Score (1-100)','cluster_id'))
#
#       load arrays back into a dataframe
#
for array in zip(data_cats_matrix,clusters,df['CustomerID'].as_matrix()):
        cluster_df = cluster_df.append({'CustomerID':array[2],'Gender':array[0][0], 'Age':array[0][1],
                                    'Annual Income (k$)':array[0][2],'Spending Score (1-100)':array[0][3]
                                    ,'cluster_id':array[1]}, ignore_index=True)

#
#
#
#       plot what we got
r=cluster_df.where(cluster_df['cluster_id']==1)
b=cluster_df.where(cluster_df['cluster_id']==2)
g=cluster_df.where(cluster_df['cluster_id']==3)
bl=cluster_df.where(cluster_df['cluster_id']==0)
        
plt.scatter(d['CustomerID'].as_matrix(),d['Annual Income (k$)'].as_matrix(), c='r')   
plt.scatter(b['CustomerID'].as_matrix(), b['Annual Income (k$)'].as_matrix(), c='b') 
plt.scatter(g['CustomerID'].as_matrix(), g['Annual Income (k$)'].as_matrix(), c='g')  
plt.scatter(bl['CustomerID'].as_matrix(), bl['Annual Income (k$)'].as_matrix(), c='black')  
 
        

    
    
#       Save results as CSV
#
cluster_df.to_csv(CSV_OUT,index=False)

