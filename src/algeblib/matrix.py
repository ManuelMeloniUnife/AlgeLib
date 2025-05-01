"""
AlgeLib - Matrix Module

This module provides operations for matrix manipulation, including basic arithmetic
operations, determinants, inverses, and other linear algebra operations.
"""

import numpy as np
from typing import Union, List, Tuple, Optional


class Matrix:
    """
    A class representing a mathematical matrix with various operations.
    
    Attributes:
        data (np.ndarray): The underlying data of the matrix.
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
    """
    
    def __init__(self, data: Union[List[List[float]], np.ndarray, 'Matrix']):
        """
        Initialize a Matrix object.
        
        Args:
            data: Initial data for the matrix. Can be a list of lists, numpy array, or another Matrix.
        
        Raises:
            ValueError: If the input data doesn't represent a valid matrix.
        """
        if isinstance(data, Matrix):
            self.data = data.data.copy()
        elif isinstance(data, np.ndarray):
            self.data = data.copy()
        elif isinstance(data, list):
            # Validate input
            if not data:
                raise ValueError("Empty matrix not allowed")
            if not all(isinstance(row, list) for row in data):
                raise ValueError("Expected a list of lists")
            if not all(len(row) == len(data[0]) for row in data):
                raise ValueError("All rows must have the same length")
            
            self.data = np.array(data, dtype=float)
        else:
            raise TypeError("Data must be a list of lists, numpy array, or Matrix")
        
        self.rows, self.cols = self.data.shape
    
    def __str__(self) -> str:
        """
        Return a string representation of the matrix.
        
        Returns:
            str: A formatted string of the matrix.
        """
        result = []
        for row in self.data:
            result.append(" ".join(f"{x:8.4f}" for x in row))
        return "\n".join(result)
    
    def __repr__(self) -> str:
        """
        Return a formal string representation of the matrix.
        
        Returns:
            str: A string that can be used to recreate the matrix.
        """
        return f"Matrix({self.data.tolist()})"
    
    def __eq__(self, other: 'Matrix') -> bool:
        """
        Check if two matrices are equal.
        
        Args:
            other: Another matrix to compare with.
            
        Returns:
            bool: True if matrices are equal, False otherwise.
        """
        if not isinstance(other, Matrix):
            return NotImplemented
        
        return np.array_equal(self.data, other.data)
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """
        Add two matrices.
        
        Args:
            other: Matrix to add to this one.
            
        Returns:
            Matrix: A new matrix representing the sum.
            
        Raises:
            ValueError: If matrices have incompatible dimensions.
        """
        if not isinstance(other, Matrix):
            return NotImplemented
        
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Cannot add matrices of size {self.rows}x{self.cols} and {other.rows}x{other.cols}")
        
        return Matrix(self.data + other.data)
    
    def __radd__(self, other) -> 'Matrix':
        """
        Support for right addition.
        
        Args:
            other: Object to add this matrix to.
            
        Returns:
            Matrix: A new matrix representing the sum.
        """
        return self.__add__(other)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """
        Subtract another matrix from this one.
        
        Args:
            other: Matrix to subtract from this one.
            
        Returns:
            Matrix: A new matrix representing the difference.
            
        Raises:
            ValueError: If matrices have incompatible dimensions.
        """
        if not isinstance(other, Matrix):
            return NotImplemented
        
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Cannot subtract matrices of size {self.rows}x{self.cols} and {other.rows}x{other.cols}")
        
        return Matrix(self.data - other.data)
    
    def __rsub__(self, other) -> 'Matrix':
        """
        Support for right subtraction.
        
        Args:
            other: Object from which to subtract this matrix.
            
        Returns:
            Matrix: A new matrix representing the difference.
        """
        if isinstance(other, Matrix):
            return other.__sub__(self)
        return NotImplemented
    
    def __mul__(self, scalar: float) -> 'Matrix':
        """
        Multiply the matrix by a scalar.
        
        Args:
            scalar: A number to multiply the matrix by.
            
        Returns:
            Matrix: A new matrix representing the product.
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        
        return Matrix(self.data * scalar)
    
    def __rmul__(self, scalar: float) -> 'Matrix':
        """
        Support for right multiplication by scalar.
        
        Args:
            scalar: A number to multiply the matrix by.
            
        Returns:
            Matrix: A new matrix representing the product.
        """
        return self.__mul__(scalar)
    
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        """
        Perform matrix multiplication.
        
        Args:
            other: Matrix to multiply with this one.
            
        Returns:
            Matrix: A new matrix representing the matrix product.
            
        Raises:
            ValueError: If matrices have incompatible dimensions for multiplication.
        """
        if not isinstance(other, Matrix):
            return NotImplemented
        
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply matrices of size {self.rows}x{self.cols} and {other.rows}x{other.cols}")
        
        return Matrix(self.data @ other.data)
    
    def transpose(self) -> 'Matrix':
        """
        Compute the transpose of the matrix.
        
        Returns:
            Matrix: The transposed matrix.
        """
        return Matrix(self.data.T)
    
    def determinant(self) -> float:
        """
        Compute the determinant of the matrix.
        
        Returns:
            float: The determinant value.
            
        Raises:
            ValueError: If the matrix is not square.
        """
        if self.rows != self.cols:
            raise ValueError("Determinant can only be calculated for square matrices")
        
        return float(np.linalg.det(self.data))
    
    def inverse(self) -> 'Matrix':
        """
        Compute the inverse of the matrix.
        
        Returns:
            Matrix: The inverse matrix.
            
        Raises:
            ValueError: If the matrix is not square or is singular.
        """
        if self.rows != self.cols:
            raise ValueError("Only square matrices can be inverted")
        
        if abs(self.determinant()) < 1e-10:
            raise ValueError("Matrix is singular and cannot be inverted")
        
        return Matrix(np.linalg.inv(self.data))
    
    def rank(self) -> int:
        """
        Compute the rank of the matrix.
        
        Returns:
            int: The rank of the matrix.
        """
        return np.linalg.matrix_rank(self.data)
    
    def eigenvalues(self) -> np.ndarray:
        """
        Compute the eigenvalues of the matrix.
        
        Returns:
            np.ndarray: Array of eigenvalues.
            
        Raises:
            ValueError: If the matrix is not square.
        """
        if self.rows != self.cols:
            raise ValueError("Eigenvalues can only be calculated for square matrices")
        
        return np.linalg.eigvals(self.data)
    
    def solve_linear_system(self, b: Union[List[float], np.ndarray, 'Matrix']) -> 'Matrix':
        """
        Solve the linear system Ax = b where A is this matrix.
        
        Args:
            b: The right-hand side vector or matrix.
            
        Returns:
            Matrix: The solution vector or matrix.
            
        Raises:
            ValueError: If the system is not solvable.
        """
        if isinstance(b, Matrix):
            b_data = b.data
        elif isinstance(b, np.ndarray):
            b_data = b
        elif isinstance(b, list):
            b_data = np.array(b, dtype=float)
        else:
            raise TypeError("b must be a list, numpy array, or Matrix")
        
        if self.rows != self.cols:
            raise ValueError("Coefficient matrix must be square")
        
        if b_data.ndim == 1 and len(b_data) != self.rows:
            raise ValueError(f"Right-hand side length ({len(b_data)}) must match matrix rows ({self.rows})")
        elif b_data.ndim > 1 and b_data.shape[0] != self.rows:
            raise ValueError(f"Right-hand side rows ({b_data.shape[0]}) must match matrix rows ({self.rows})")
        
        try:
            solution = np.linalg.solve(self.data, b_data)
            return Matrix(solution)
        except np.linalg.LinAlgError:
            raise ValueError("The system is singular and cannot be solved")
    
    def lu_decomposition(self) -> Tuple['Matrix', 'Matrix']:
        """
        Perform LU decomposition of the matrix.
        
        Returns:
            Tuple[Matrix, Matrix]: The lower and upper triangular matrices.
            
        Raises:
            ValueError: If the matrix is not square.
        """
        if self.rows != self.cols:
            raise ValueError("LU decomposition requires a square matrix")
        
        # NumPy's LU decomposition includes a permutation matrix P
        # For simplicity, we're ignoring P here
        p, l, u = np.linalg.lu(self.data)
        return Matrix(l), Matrix(u)
    
    def qr_decomposition(self) -> Tuple['Matrix', 'Matrix']:
        """
        Perform QR decomposition of the matrix.
        
        Returns:
            Tuple[Matrix, Matrix]: The orthogonal matrix Q and upper triangular matrix R.
        """
        q, r = np.linalg.qr(self.data)
        return Matrix(q), Matrix(r)
    
    @classmethod
    def identity(cls, size: int) -> 'Matrix':
        """
        Create an identity matrix of the given size.
        
        Args:
            size: The size of the square identity matrix.
            
        Returns:
            Matrix: An identity matrix.
            
        Raises:
            ValueError: If size is not positive.
        """
        if size <= 0:
            raise ValueError("Size must be positive")
        
        return cls(np.eye(size))
    
    @classmethod
    def zeros(cls, rows: int, cols: Optional[int] = None) -> 'Matrix':
        """
        Create a matrix filled with zeros.
        
        Args:
            rows: Number of rows.
            cols: Number of columns. If None, creates a square matrix.
            
        Returns:
            Matrix: A matrix filled with zeros.
            
        Raises:
            ValueError: If dimensions are not positive.
        """
        if rows <= 0:
            raise ValueError("Number of rows must be positive")
        
        if cols is None:
            cols = rows
        elif cols <= 0:
            raise ValueError("Number of columns must be positive")
        
        return cls(np.zeros((rows, cols)))
    
    @classmethod
    def ones(cls, rows: int, cols: Optional[int] = None) -> 'Matrix':
        """
        Create a matrix filled with ones.
        
        Args:
            rows: Number of rows.
            cols: Number of columns. If None, creates a square matrix.
            
        Returns:
            Matrix: A matrix filled with ones.
            
        Raises:
            ValueError: If dimensions are not positive.
        """
        if rows <= 0:
            raise ValueError("Number of rows must be positive")
        
        if cols is None:
            cols = rows
        elif cols <= 0:
            raise ValueError("Number of columns must be positive")
        
        return cls(np.ones((rows, cols)))
    
    @classmethod
    def random(cls, rows: int, cols: Optional[int] = None) -> 'Matrix':
        """
        Create a matrix filled with random values between 0 and 1.
        
        Args:
            rows: Number of rows.
            cols: Number of columns. If None, creates a square matrix.
            
        Returns:
            Matrix: A matrix filled with random values.
            
        Raises:
            ValueError: If dimensions are not positive.
        """
        if rows <= 0:
            raise ValueError("Number of rows must be positive")
        
        if cols is None:
            cols = rows
        elif cols <= 0:
            raise ValueError("Number of columns must be positive")
        
        return cls(np.random.rand(rows, cols))