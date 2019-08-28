import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def generate_point_set(n_list, random_seed = 42):
    """
    Generate len(n) random 2D points and return the list of points (2D arrays)
    The number of points in the sets are in n_list
    The range of the points coordinates are -1 to 1
    """
    points_list = []
    np.random.seed(random_seed)
    for n in n_list:
        points_list.append(np.random.rand(n,2) * 2 - 1)
    return points_list




def plot_point_sets(points_list, Div_mat, divers):#, diversity_measure_list, diversity_measure_name_list):
    """
    Plot the points in the points list with the diversity measurement
    :param points_list: the list of points
    :param diversity_measure_name_list: The list of diversity measures
    :param diversity_measure_list: The list of names of diversity measures to show at legend
    """

    for cnt, points in enumerate(points_list):
        f = plt.figure()
        plt.scatter(points[:,0],points[:,1])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(str(cnt))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        f.savefig('{}.png'.format(cnt))

def diversity_list(points_list, diversity_measures):
    #Creat the 2D matrix for diversity measure
    Div_mat = np.zeros(len(points_list), len(diversity_measures))

    for i, points in enumerate(points_list):
        for j,div_measure in enumerate(diversity_measures):
            Div_mat[i,j] = div_measure(points)

    return Div_mat

def variance(points):
    if (len(points) < 2):
        return 0
    else:
        return np.sum(np.var(points, axis = 0))

def mean_NN_dist(points):
    nbrs = NearestNeighbors(n_neighbors = 2).fit(points)
    distance, indices = nbrs.kneighbors(points)
    return np.sum(distance) / len(points) 

def sum_NN_dist(points):
    return mean_NN_dist(points) *  len(points)

def mean_dist(points):
    nbrs = NearestNeighbors(n_neighbors = len(points)).fit(points)
    distance, indices = nbrs.kneighbors(points)
    return np.sum(distance) / (len(points) - 1) / len(points)

def sum_dist(points):
    return mean_dist(points) * len(points)

if __name__ == '__main__':
    points = generate_point_set([1,2,2,3,3,4,4,5,6])
    #plot_point_sets(points)
    diversity_measures = [variance, mean_NN_dist, mean_dist, sum_NN_dist, sum_dist]
    diversity_names = ['variance', 'mean_NN_dist', 'mean_dist', 'sum_NN_dist', 'sum_dist']
   
    
    
    
    points = [[0,0],[1,1],[0,1],[1,0]]
    print(variance(points))
    print(mean_NN_dist(points))
    print(mean_dist(points))
    
    points = [[0,0],[1,1],[0,1],[1,0],[0.5,0.5]]
    print(variance(points))
    print(mean_NN_dist(points))
    print(mean_dist(points))
