import numpy as np

#A = np.random.randint(0, 10, (4, 4))
A = np.array([[1, 2, 4], [2, 4, 8], [2, 6, 13]])
P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
A = A.astype('float32')
#print(P @ A)

class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape
        self.rows = matrix.shape[0]
        self.cols = matrix.shape[1]

    def LU(self, matrix, pivoting = True):
        """
        This function reduces a matrix to its row echelon form using Gaussian elimination.
        """
        #decomposition follows the diagonal:
        row_exchanges = 0
        i, j = 0, 0
        P = np.eye(matrix.shape[0])
        A = matrix.copy()
        L = np.eye(matrix.shape[0])
        while i < matrix.shape[0] and j < matrix.shape[1]:
            #find the pivot
            #print(A)
            if pivoting:
                pivot = np.argmax(np.abs(A[i:, j])) + i
                if A[pivot, j] == 0: 
                    i += 1
                    j += 1
                    continue
            else:
                pivot = i        
            #swap rows:
            if i != pivot:
                row_0 = A[i].copy()
                A[i] = A[pivot]
                A[pivot] = row_0
                #swap the permutation matrix:
                row_0 = P[i].copy()
                P[i] = P[pivot]
                P[pivot] = row_0   
                row_exchanges += 1
            #eliminate rows bellow:
            factor = A[i + 1:, j] / A[i, j]
            L[(i + 1):, j] = factor
            A[i + 1:, j:] = A[i + 1:, j:] - factor[:, np.newaxis] * A[i, j:]
            i += 1
            j += 1
        self.P, self.U, self.L = P, A, L
        self.sgn = 1 if row_exchanges % 2 == 0 or row_exchanges == 0 else -1
        rank = 0
        for row in range(self.U.shape[0]):
            if np.count_nonzero(self.U[row]) != 0:
               rank += 1
        self.rank = rank
        self.nullity = self.U.shape[0] - rank

    def determinant(self):
        """
        This function returns the determinant of the matrix.
        """
        self.LU(self.matrix)
        udiag = np.diagonal(self.U)
        ldiag = np.diagonal(self.L)
        det = 1
        for i in range(len(udiag)):
            det *= udiag[i]
        for i in range(len(ldiag)):
            det *= ldiag[i]
        self.det = det
        return det * self.sgn

A = np.array([[-3, 2, -1], [-2, 0, -3],[-1, 3, 2]]) 
A = A.astype('float64')
mat = Matrix(A)
mat.determinant()
print(np.linalg.det(A))
print(mat.U)
print(mat.P @ A)
print(mat.det)