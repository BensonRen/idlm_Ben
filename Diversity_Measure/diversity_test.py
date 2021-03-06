import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from pandas.plotting import table
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def generate_point_set(n_list, random_seed = 42):
    """
    Generate len(n) random 2D points and return the list of points (2D arrays)
    The number of points in the sets are in n_list
    The range of the points coordinates are -1 to 1
    """
    points_list = []
    np.random.seed(random_seed)
    for n in n_list:
        points_list.append(np.random.rand(n,2))
    return points_list

def plot_point_sets(points_list, Div_mat, diversity_names):#, diversity_measure_list, diversity_measure_name_list):
    """
    Plot the points in the points list with the diversity measurement
    :param points_list: the list of points
    :param diversity_measure_name_list: The list of diversity measures
    :param diversity_measure_list: The list of names of diversity measures to show at legend
    """
    df = pd.DataFrame(Div_mat)#, columns = diversity_names)
    df.columns = diversity_names
    for cnt, points in enumerate(points_list):
        #df = pd.DataFrame(np.transpose(Div_mat[cnt,:]))#, columns = diversity_names)
        #df.info()
        #df = df.T
        #df.columns = diversity_names
        f = plt.figure()
        ax = plt.gca()
        table(ax, np.round(df.iloc[cnt,:],2), loc = 'upper right', colWidths=[0.05])
        plt.scatter(points[:,0],points[:,1])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(str(cnt))
        plt.xlim(0,1)
        plt.ylim(0,1)
        f.savefig('{}.png'.format(cnt))
        
def get_div_mat(points_list, diversity_measures):
    #Creat the 2D matrix for diversity measure
    Div_mat = np.zeros([len(points_list), len(diversity_measures)])

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
    if (len(points) < 2):
        return 0
    nbrs = NearestNeighbors(n_neighbors = 2).fit(points)
    distance, indices = nbrs.kneighbors(points)
    return np.sum(distance) / len(points) 

def sum_NN_dist(points):
    return mean_NN_dist(points) *  len(points)

def mean_dist(points):
    if (len(points) < 2):
        return 0
    nbrs = NearestNeighbors(n_neighbors = len(points)).fit(points)
    distance, indices = nbrs.kneighbors(points)
    return np.sum(distance) / (len(points) - 1) / len(points)

def sum_dist(points):
    return mean_dist(points) * len(points)

def NNmulSum_dist(points):
    if (len(points) < 2):
        return 0
    metric = 0
    nbrs = NearestNeighbors(n_neighbors = len(points)).fit(points)
    distance, indices = nbrs.kneighbors(points)
    for i in range(len(distance)):
        point_contribution = np.sum(distance[i,:]) * distance[i,1] #The total distance times the nearest neighbor
        metric += point_contribution
    return metric

def MST2(points):   #The minimum spanning tree length of the points
    if (len(points) < 2):
        return 0
    distance_matrix_points = distance_matrix(points, points, p = 2)
    csr_mat = csr_matrix(distance_matrix_points)
    Tree = minimum_spanning_tree(csr_mat)
    return np.sum(Tree.toarray().astype(float))

def MST1(points):   #The minimum spanning tree length of the points
    if (len(points) < 2):
        return 0
    distance_matrix_points = distance_matrix(points, points, p = 1)
    csr_mat = csr_matrix(distance_matrix_points)
    Tree = minimum_spanning_tree(csr_mat)
    return np.sum(Tree.toarray().astype(float))

def MST3(points):   #The minimum spanning tree length of the points
    if (len(points) < 2):
        return 0
    distance_matrix_points = distance_matrix(points, points, p = 3)
    csr_mat = csr_matrix(distance_matrix_points)
    Tree = minimum_spanning_tree(csr_mat)
    return np.sum(Tree.toarray().astype(float))

def MST4(points):   #The minimum spanning tree length of the points
    if (len(points) < 2):
        return 0
    distance_matrix_points = distance_matrix(points, points, p = 4)
    csr_mat = csr_matrix(distance_matrix_points)
    Tree = minimum_spanning_tree(csr_mat)
    return np.sum(Tree.toarray().astype(float))

def heat_maps_for_metrics(points, diversity_measures, diversity_names, save_name = ''):
    """
    The function which plots the color map for different diversity measurements
    """
    X = np.arange(0,1,0.01)
    Y = np.arange(0,1,0.01)
    X_grid,Y_grid = np.meshgrid(X,Y)
    #Z = np.arrays(X_grid)
    grid_h, grid_w = np.shape(X_grid)
    Z = np.zeros([grid_h, grid_w])
    h,w = np.shape(points)
    point_new = np.zeros([h+1,w])
    assert w==2
    point_new[0:h,:] = points
    point_grid = [[[] for col in range(grid_w)] for row in range(grid_h)]
    for i in range(grid_h):
        for j in range(grid_w):
            point_new[h,0] = X[i]
            point_new[h,1] = Y[j]
            point_grid[i][j] = np.array(point_new)
            #print('point new at {},{} is {}'.format(i,j,point_new))

    for cnt, (div_measure, div_name) in enumerate(zip(diversity_measures, diversity_names)):
        f = plt.figure()
        ax = plt.gca()
        current_metric_value = div_measure(points)
        print("current metric value",current_metric_value)
        for i in range(grid_h):
            for j in range(grid_w):
                #print('for point {}, {}, the point is {}'.format(i,j,point_grid[i][j]))
                Z[i][j] = div_measure(point_grid[i][j])
        #print(Z)
        #plt.scatter(points[:,0], points[:,1],color = 'k',linewidths = 15,label = 'Anchor points')
        Zmax = np.max(Z)
        Zmin = np.min(Z)
        C = ax.pcolormesh(X_grid, Y_grid, Z, cmap = 'jet')
        plt.scatter(points[:,1], points[:,0],color = 'k',linewidths = 1,label = 'Anchor {}'.format(current_metric_value))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('color map for {} metric'.format(div_name))
        plt.xlim(0,1)
        plt.ylim(0,1)
        fb = f.colorbar(C, ax = ax)
        line_pos = (current_metric_value - Zmin)/ (Zmax - Zmin)
        fb.ax.plot([0,1],[line_pos, line_pos], color = 'w', linewidth = 3,label = 'current value')
        plt.legend()
        f.savefig(save_name + div_name + 'heatmap')
    plt.close('all')
if __name__ == '__main__':
    points = generate_point_set([1,2,3,4,5,6,7])
    #plot_point_sets(points)
    #diversity_measures = [variance, mean_NN_dist, mean_dist, sum_NN_dist, sum_dist, NNmulSum_dist, MST]
    #diversity_names = ['variance', 'mean_NN_dist', 'mean_dist', 'sum_NN_dist', 'sum_dist','NN*sum_dist','MST']
    diversity_measures = [ MST1,MST2,MST3,MST4]
    diversity_names = ['MST1','MST2','MST3','MST4']
    #div_mat = get_div_mat(points, diversity_measures)
    #plot_point_sets(points, div_mat, diversity_names)
    
    for cnt, point in enumerate(points):
        heat_maps_for_metrics(point, diversity_measures, diversity_names, save_name = '{}_point_example'.format(cnt + 1))
         
    #points = np.array([[0.25,0.5],[0.75,0.5]])
    #heat_maps_for_metrics(points, diversity_measures, diversity_names, save_name = '2_point_example')
    #points = np.array([[0.1,0.5]])
    
    #points = np.array([[0.25,0.5],[0.75,0.5],[0.5,0.8]])
    #heat_maps_for_metrics(points, diversity_measures, diversity_names, save_name = '3_point_example')
    points = [[0,0],[1,1],[0,1],[1,0],[0.5,0.5]]
    print(variance(points))
    print(mean_NN_dist(points))
    print(mean_dist(points))
