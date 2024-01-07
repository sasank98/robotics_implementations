import numpy as np
import pandas as pd
from Transformation import *
from Point_cloud import Point_cloud, write_ply
from Generalized_ICP import *
import plotly.graph_objs as go

if __name__ == '__main__':
    
    dfx = pd.read_csv("./pclX.txt" , sep=" ", header = None)
    X = dfx.to_numpy(dtype=float)
    # print(X.shape)
    dfy = pd.read_csv('./pclY.txt', sep=" ", header = None)
    Y=dfy.to_numpy(dtype=float)

    data = Point_cloud()
    ref = Point_cloud()

    n_iter = 50
    thresholds = [0.07,0.08,0.09,0.1,0.11,0.12]
    methods = ["point2point","point2plane","plane2plane"]
    data.init_from_points(X)
    ref.init_from_points(Y)
    last_rms = np.zeros((len(thresholds),n_iter,len(methods)))

    R, T, rms_list = ICP(data,ref, method = methods[0], exclusion_radius = thresholds[0] ,sampling_limit = None, verbose = True)
    # print(R, T)

    corr_x = np.dot(X, R.T)+ T
    fig = go.Figure()
    # fig.add_trace(go.Scatter3d(name= 'X', mode = 'markers', x=X[:,0], 
    #     y=X[:,1], z=X[:,2], marker=dict(color='rgb(256,0,0)', size=1)))
    fig.add_trace(go.Scatter3d(name= 'X', mode = 'markers', x=Y[:,0], 
        y=Y[:,1], z=Y[:,2], marker=dict(color='rgb(0,0,256)', size=1)))
    fig.add_trace(go.Scatter3d(name= 'corrected X', mode = 'markers', x=corr_x[:,0], y=corr_x[:,1], 
        z=corr_x[:,2], marker=dict(color='rgb(0,256,0)', size=1)))
    fig.show()
