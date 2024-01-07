# from utils.ply import read_ply, write_ply
import numpy as np
from sklearn.neighbors import KDTree
import sys

class Point_cloud:
    def __init__(self):

        self.all_eigenvalues = None
        self.all_eigenvectors = None
        self.n = 0
        self.kdtree = None
        self.points = None
        self.nn = None

    def save(self,path):
        write_ply(path,self.points,["x","y","z"])

    def init_from_ply(self,path):
        tmp = read_ply(path)
        self.points = np.vstack((tmp['x'],tmp['y'],tmp['z'])).T
        self._init()


    def init_from_transfo(self, initial, R = None,t = None):

        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)

        self.points = initial.points @ R.T + t
        self._init()

    def init_from_points(self,points):

        self.points = points.copy()
        self._init()

    def _init(self):

        self.kdtree = KDTree(self.points)
        self.n  = self.points.shape[0] ### CONFIRM
        self.all_eigenvalues = None
        self.all_eigenvectors = None
        self.nn = None

    def transform(self,R,T):

        self.points = self.points @ R.T + T
        if not self.all_eigenvectors is None:
            self.all_eigenvectors = self.all_eigenvectors @ R.T

    def neighborhood_PCA(self, radius = 0.005):

        if self.nn is None:
            self.nn = self.kdtree.query_radius(self.points,r = radius, return_distance = False)

        all_eigenvalues = np.zeros((self.n, 3))
        all_eigenvectors = np.zeros((self.n, 3, 3))

        for i in range(self.n):
            if len(self.nn[i]) < 3:
                all_eigenvalues[i], all_eigenvectors[i] = (np.array([0,0,0]),np.eye(3))
            else:
                all_eigenvalues[i], all_eigenvectors[i] = np.linalg.eigh(np.cov(self.points[self.nn[i]].T))

        return all_eigenvalues, all_eigenvectors

    def get_eigenvectors(self, radius = 0.005):

        if self.all_eigenvectors is None:
            self.all_eigenvalues,self.all_eigenvectors = self.neighborhood_PCA(radius)
        return self.all_eigenvectors

    def get_projection_matrix_point2plane(self, indexes = None):

        if indexes is None:
            indexes = np.arange(self.n)

        all_eigenvectors = self.get_eigenvectors()
        normals = all_eigenvectors[:,:,0]
        normals = normals / np.linalg.norm(normals, axis = 1, keepdims = True)
        return np.array([normals[i,:,None]*normals[i,None,:] for i in indexes])

    def get_covariance_matrices_plane2plane(self, epsilon  = 1e-3,indexes = None):

        if indexes is None:
            indexes = np.arange(self.n)
        d = 3
        new_n = indexes.shape[0]
        cov_mat = np.zeros((new_n,d,d))
        all_eigenvectors = self.get_eigenvectors()
        dz_cov_mat = np.eye(d)
        dz_cov_mat[0,0] = epsilon
        for i in range(new_n):
            U = all_eigenvectors[indexes[i]]
            cov_mat[i,:,:] = U @ dz_cov_mat @ U.T

        return cov_mat
    

valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}

ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'b1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])
    
def read_ply(filename):

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # Parse header
        num_points, properties = parse_header(plyfile, ext)

        # Get data
        data = np.fromfile(plyfile, dtype=properties, count=num_points)


    return data


def header_properties(field_list, field_names):
    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties

def write_ply(filename, field_list, field_names):

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field is None:
            print('WRITE_PLY ERROR: a field is None')
            return False
        elif field.ndim > 2:
            print('WRITE_PLY ERROR: a field have more than 2 dimensions')
            return False
        elif field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)


    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

    return True
