"""
AlgeLib - Polynomial Module

This module provides operations for polynomial manipulation, including basic arithmetic
operations, roots, derivatives, integrals, and other algebraic operations.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Sequence


class Polynomial:
    """
    A class representing a mathematical polynomial with various operations.
    
    Attributes:
        coefficients (np.ndarray): The coefficients of the polynomial in ascending order of degree.
                        e.g., [3, 2, 1] represents 3 + 2x + 1x²
    """
    
    def __init__(self, coefficients: Union[List[float], np.ndarray, 'Polynomial']):
        """
        Initialize a Polynomial object.
        
        Args:
            coefficients: Coefficients of the polynomial in ascending order of degree.
                        Can be a list, numpy array, or another Polynomial.
        """
        if isinstance(coefficients, Polynomial):
            self.coefficients = coefficients.coefficients.copy()
        elif isinstance(coefficients, np.ndarray):
            self.coefficients = coefficients.copy()
        elif isinstance(coefficients, (list, tuple)):
            self.coefficients = np.array(coefficients, dtype=float)
        else:
            raise TypeError("Coefficients must be a list, numpy array, or Polynomial")
        
        # Remove leading zeros (highest degree terms with coefficient 0)
        self._trim()
    
    def _trim(self) -> None:
        """
        Remove leading zeros from the coefficients array.
        """
        if len(self.coefficients) > 1:
            idx = len(self.coefficients) - 1
            while idx >= 0 and abs(self.coefficients[idx]) < 1e-10:
                idx -= 1
            
            # Keep at least one coefficient (the constant term)
            if idx < 0:
                self.coefficients = np.array([0.0])
            else:
                self.coefficients = self.coefficients[:idx + 1]
    
    def degree(self) -> int:
        """
        Get the degree of the polynomial.
        
        Returns:
            int: The degree of the polynomial.
        """
        # The degree is one less than the length of the coefficients array,
        # except for the zero polynomial (which has degree -1 by convention)
        if len(self.coefficients) == 1 and abs(self.coefficients[0]) < 1e-10:
            return -1
        return len(self.coefficients) - 1
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the polynomial.
        
        Returns:
            str: A string representation of the polynomial.
        """
        if self.degree() == -1:
            return "0"
        
        terms = []
        for i, coef in enumerate(self.coefficients):
            if abs(coef) < 1e-10:  # Skip zero coefficients
                continue
                
            # Format the coefficient
            if i == 0:  # Constant term
                term = f"{coef:.4f}".rstrip('0').rstrip('.')
            else:
                # Skip coefficient if it's 1 or -1, unless it's just "1" or "-1"
                if abs(abs(coef) - 1) < 1e-10:
                    if coef > 0:
                        term = ""
                    else:
                        term = "-"
                else:
                    term = f"{coef:.4f}".rstrip('0').rstrip('.')
                
                # Add the variable with appropriate power
                if i == 1:  # Linear term
                    term += "x"
                else:  # Higher power
                    term += f"x^{i}"
            
            # Add plus sign for positive terms after the first term
            if coef > 0 and i > 0:
                term = f"+ {term}"
            elif coef < 0 and i > 0:
                term = f"- {term.lstrip('-')}"
                
            terms.append(term)
        
        # Join all terms and return the resulting string
        polynomial_str = " ".join(terms)
        
        # Fix the string representation if the first term is negative
        if self.coefficients[0] < 0:
            polynomial_str = "-" + polynomial_str.lstrip("- ")
            
        return polynomial_str
    
    def __repr__(self) -> str:
        """
        Return a formal string representation of the polynomial.
        
        Returns:
            str: A string that can be used to recreate the polynomial.
        """
        return f"Polynomial({self.coefficients.tolist()})"
    
    def __eq__(self, other: 'Polynomial') -> bool:
        """
        Check if two polynomials are equal.
        
        Args:
            other: Another polynomial to compare with.
            
        Returns:
            bool: True if polynomials are equal, False otherwise.
        """
        if not isinstance(other, Polynomial):
            return NotImplemented
        
        # Polynomials are equal if their coefficients are equal
        if len(self.coefficients) != len(other.coefficients):
            return False
        
        return np.allclose(self.coefficients, other.coefficients)
    
    def __add__(self, other: Union['Polynomial', float, int]) -> 'Polynomial':
        """
        Add another polynomial or a constant to this polynomial.
        
        Args:
            other: Polynomial or constant to add.
            
        Returns:
            Polynomial: A new polynomial representing the sum.
        """
        if isinstance(other, (int, float)):
            # Adding a constant is equivalent to adding a constant polynomial
            result = self.coefficients.copy()
            result[0] += other
            return Polynomial(result)
        
        if not isinstance(other, Polynomial):
            return NotImplemented
        
        # Determine the degree of the resulting polynomial
        max_length = max(len(self.coefficients), len(other.coefficients))
        result = np.zeros(max_length)
        
        # Add the coefficients of the same degree
        result[:len(self.coefficients)] += self.coefficients
        result[:len(other.coefficients)] += other.coefficients
        
        return Polynomial(result)
    
    def __radd__(self, other: Union[float, int]) -> 'Polynomial':
        """
        Support for right addition.
        
        Args:
            other: Object to add this polynomial to.
            
        Returns:
            Polynomial: A new polynomial representing the sum.
        """
        return self.__add__(other)
    
    def __sub__(self, other: Union['Polynomial', float, int]) -> 'Polynomial':
        """
        Subtract another polynomial or a constant from this polynomial.
        
        Args:
            other: Polynomial or constant to subtract.
            
        Returns:
            Polynomial: A new polynomial representing the difference.
        """
        if isinstance(other, (int, float)):
            # Subtracting a constant is equivalent to subtracting a constant polynomial
            result = self.coefficients.copy()
            result[0] -= other
            return Polynomial(result)
        
        if not isinstance(other, Polynomial):
            return NotImplemented
        
        # Determine the degree of the resulting polynomial
        max_length = max(len(self.coefficients), len(other.coefficients))
        result = np.zeros(max_length)
        
        # Add the coefficients of the same degree
        result[:len(self.coefficients)] += self.coefficients
        result[:len(other.coefficients)] -= other.coefficients
        
        return Polynomial(result)
    
    def __rsub__(self, other: Union[float, int]) -> 'Polynomial':
        """
        Support for right subtraction.
        
        Args:
            other: Object from which to subtract this polynomial.
            
        Returns:
            Polynomial: A new polynomial representing the difference.
        """
        if isinstance(other, (int, float)):
            # Subtracting polynomial from constant
            result = -self.coefficients.copy()
            result[0] += other
            return Polynomial(result)
        return NotImplemented
    
    def __neg__(self) -> 'Polynomial':
        """
        Negate the polynomial.
        
        Returns:
            Polynomial: A new polynomial with all coefficients negated.
        """
        return Polynomial(-self.coefficients)
    
    def __mul__(self, other: Union['Polynomial', float, int]) -> 'Polynomial':
        """
        Multiply this polynomial by another polynomial or a constant.
        
        Args:
            other: Polynomial or constant to multiply by.
            
        Returns:
            Polynomial: A new polynomial representing the product.
        """
        if isinstance(other, (int, float)):
            # Multiplication by a constant
            return Polynomial(other * self.coefficients)
        
        if not isinstance(other, Polynomial):
            return NotImplemented
        
        # For polynomial multiplication, the degree of the result is the sum of the degrees
        result_degree = len(self.coefficients) + len(other.coefficients) - 1
        result = np.zeros(result_degree)
        
        # Multiply each term of this polynomial by each term of the other polynomial
        for i in range(len(self.coefficients)):
            for j in range(len(other.coefficients)):
                result[i + j] += self.coefficients[i] * other.coefficients[j]
        
        return Polynomial(result)
    
    def __rmul__(self, other: Union[float, int]) -> 'Polynomial':
        """
        Support for right multiplication.
        
        Args:
            other: Object to multiply this polynomial by.
            
        Returns:
            Polynomial: A new polynomial representing the product.
        """
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Polynomial', float, int]) -> 'Polynomial':
        """
        Divide this polynomial by another polynomial or a constant.
        
        Args:
            other: Polynomial or constant to divide by.
            
        Returns:
            Polynomial: A new polynomial representing the quotient.
            
        Raises:
            ValueError: If dividing by zero or by a polynomial with degree 0.
            NotImplementedError: If dividing by a polynomial with degree > 0.
        """
        if isinstance(other, (int, float)):
            if abs(other) < 1e-10:
                raise ValueError("Division by zero")
            return Polynomial(self.coefficients / other)
        
        if not isinstance(other, Polynomial):
            return NotImplemented
        
        # For now, we'll only implement division by a constant polynomial
        if other.degree() == 0:
            return self / other.coefficients[0]
        
        # For polynomial division (with remainder), use quotient_remainder method
        quotient, _ = self.quotient_remainder(other)
        return quotient
    
    def __call__(self, x: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the polynomial at point x.
        
        Args:
            x: Value or array of values at which to evaluate the polynomial.
            
        Returns:
            The value of the polynomial at x.
        """
        return np.polyval(np.flip(self.coefficients), x)
    
    def derivative(self) -> 'Polynomial':
        """
        Compute the derivative of the polynomial.
        
        Returns:
            Polynomial: The derivative polynomial.
        """
        if self.degree() <= 0:
            return Polynomial([0])
        
        # Multiply each coefficient by its degree
        deriv_coeffs = np.array([i * c for i, c in enumerate(self.coefficients[1:], 1)])
        return Polynomial(deriv_coeffs)
    
    def integral(self, constant: float = 0) -> 'Polynomial':
        """
        Compute the indefinite integral of the polynomial.
        
        Args:
            constant: Integration constant.
            
        Returns:
            Polynomial: The integral polynomial.
        """
        # Divide each coefficient by its new degree
        integral_coeffs = np.zeros(len(self.coefficients) + 1)
        integral_coeffs[0] = constant
        integral_coeffs[1:] = np.array([c / i for i, c in enumerate(self.coefficients, 1)])
        
        return Polynomial(integral_coeffs)
    
    def definite_integral(self, a: float, b: float) -> float:
        """
        Compute the definite integral of the polynomial from a to b.
        
        Args:
            a: Lower bound of integration.
            b: Upper bound of integration.
            
        Returns:
            float: The value of the definite integral.
        """
        # Compute the indefinite integral
        integral = self.integral()
        
        # Evaluate the indefinite integral at the bounds and subtract
        return integral(b) - integral(a)
    
    def roots(self) -> np.ndarray:
        """
        Compute the roots of the polynomial.
        
        Returns:
            np.ndarray: Array of roots.
            
        Raises:
            ValueError: If the polynomial is constant.
        """
        if self.degree() <= 0:
            raise ValueError("Cannot find roots of a constant polynomial unless it's zero")
        
        # np.roots expects coefficients in descending order
        return np.roots(np.flip(self.coefficients))
    
    def quotient_remainder(self, divisor: 'Polynomial') -> Tuple['Polynomial', 'Polynomial']:
        """
        Divide this polynomial by another polynomial and return quotient and remainder.
        
        Args:
            divisor: The polynomial to divide by.
            
        Returns:
            Tuple[Polynomial, Polynomial]: The quotient and remainder polynomials.
            
        Raises:
            ValueError: If divisor is the zero polynomial.
        """
        if divisor.degree() == -1:  # Zero polynomial
            raise ValueError("Division by zero polynomial")
        
        # If dividend degree is less than divisor degree, quotient is 0 and remainder is the dividend
        if self.degree() < divisor.degree():
            return Polynomial([0]), Polynomial(self.coefficients.copy())
        
        # Copy the dividend coefficients in reverse order (highest degree first)
        dividend = np.flip(self.coefficients.copy())
        divisor_coef = np.flip(divisor.coefficients.copy())
        
        # Initialize quotient array with zeros
        quotient_degree = self.degree() - divisor.degree()
        quotient = np.zeros(quotient_degree + 1)
        
        # Polynomial long division
        for i in range(quotient_degree + 1):
            # Calculate the next quotient coefficient
            quotient[i] = dividend[i] / divisor_coef[0]
            
            # Update the dividend: subtract divisor * quotient[i]
            for j in range(len(divisor_coef)):
                dividend[i + j] -= quotient[i] * divisor_coef[j]
        
        # The remainder is what's left of the dividend (truncated to the right size)
        remainder_degree = divisor.degree() - 1
        remainder = np.flip(dividend[-(remainder_degree + 1):]) if remainder_degree >= 0 else np.array([0])
        
        # Return the quotient and remainder as Polynomial objects
        return Polynomial(np.flip(quotient)), Polynomial(remainder)
    
    def gcd(self, other: 'Polynomial') -> 'Polynomial':
        """
        Compute the greatest common divisor of this polynomial and another.
        
        Args:
            other: Another polynomial.
            
        Returns:
            Polynomial: The greatest common divisor.
        """
        # If either polynomial is zero, the GCD is the other polynomial
        if self.degree() == -1:
            return Polynomial(other.coefficients / other.coefficients[-1])  # Normalize
        if other.degree() == -1:
            return Polynomial(self.coefficients / self.coefficients[-1])  # Normalize
        
        # Euclidean algorithm for GCD
        a = Polynomial(self.coefficients.copy())
        b = Polynomial(other.coefficients.copy())
        
        while b.degree() >= 0:
            _, r = a.quotient_remainder(b)
            a, b = b, r
        
        # Normalize the GCD to have leading coefficient 1
        return Polynomial(a.coefficients / a.coefficients[-1])
    
    def lcm(self, other: 'Polynomial') -> 'Polynomial':
        """
        Compute the least common multiple of this polynomial and another.
        
        Args:
            other: Another polynomial.
            
        Returns:
            Polynomial: The least common multiple.
            
        Raises:
            ValueError: If either polynomial is zero.
        """
        if self.degree() == -1 or other.degree() == -1:
            raise ValueError("LCM is not defined for zero polynomials")
        
        # LCM(a, b) = (a * b) / GCD(a, b)
        gcd = self.gcd(other)
        prod = self * other
        quotient, remainder = prod.quotient_remainder(gcd)
        
        # The remainder should be zero
        if remainder.degree() >= 0:
            raise ValueError("Failed to compute LCM")
        
        # Normalize the LCM to have leading coefficient 1
        return Polynomial(quotient.coefficients / quotient.coefficients[-1])
    
    @classmethod
    def from_roots(cls, roots: Sequence[complex]) -> 'Polynomial':
        """
        Create a polynomial from its roots.
        
        Args:
            roots: Sequence of roots.
            
        Returns:
            Polynomial: A polynomial with the given roots.
        """
        # For a polynomial with roots r₁, r₂, ..., rₙ, the formula is:
        # P(x) = (x - r₁)(x - r₂)...(x - rₙ)
        
        if not roots:
            return cls([1])  # Empty roots list returns the polynomial 1
        
        # Start with P(x) = 1
        result = cls([1])
        
        # Multiply by (x - r) for each root r
        for root in roots:
            # (x - r) is represented as the polynomial [-r, 1]
            factor = cls([-root, 1])
            result = result * factor
            
        # If roots has only real values but with floating-point errors,
        # ensure the result has real coefficients
        if all(abs(root.imag) < 1e-10 for root in roots):
            result.coefficients = np.real(result.coefficients)
            
        return result
    
    @classmethod
    def monomial(cls, degree: int, coefficient: float = 1.0) -> 'Polynomial':
        """
        Create a monomial of the given degree.
        
        Args:
            degree: The degree of the monomial.
            coefficient: The coefficient of the monomial (default 1.0).
            
        Returns:
            Polynomial: A monomial of the form coefficient * x^degree.
            
        Raises:
            ValueError: If degree is negative.
        """
        if degree < 0:
            raise ValueError("Degree must be non-negative")
            
        coeffs = np.zeros(degree + 1)
        coeffs[degree] = coefficient
        
        return cls(coeffs)
    
    @classmethod
    def from_string(cls, poly_str: str) -> 'Polynomial':
        """
        Create a polynomial from a string representation.
        
        Args:
            poly_str: String representation of a polynomial (e.g., "3x^2 + 2x - 1").
            
        Returns:
            Polynomial: The polynomial represented by the string.
            
        Raises:
            ValueError: If the string is not a valid polynomial.
        """
        import re
        
        # Remove spaces
        poly_str = poly_str.replace(" ", "")
        
        # Replace '-' with '+-' for easier splitting, but avoid double +
        poly_str = poly_str.replace("-", "+-").replace("++", "+")
        
        # If the string starts with '+', remove it
        if poly_str.startswith("+"):
            poly_str = poly_str[1:]
            
        # Split the string by '+' to get individual terms
        terms = poly_str.split("+")
        
        # Initialize an empty dictionary to store coefficients
        coeffs_dict = {}
        
        # Regular expression patterns to match different types of terms
        const_pattern = r'^(-?\d*\.?\d*)$'
        linear_pattern = r'^(-?\d*\.?\d*)x$'
        power_pattern = r'^(-?\d*\.?\d*)x\^(\d+)$'

        
        for term in terms:
            if not term:  # Skip empty terms
                continue
                
            # Try to match the term with one of the patterns
            const_match = re.match(const_pattern, term)
            linear_match = re.match(linear_pattern, term)
            power_match = re.match(power_pattern, term)
            
            if const_match:
                # Constant term
                coef = float(const_match.group(1) or 1)
                coeffs_dict[0] = coeffs_dict.get(0, 0) + coef
            elif linear_match:
                # Linear term (x term)
                coef_str = linear_match.group(1)
                if coef_str in ("", "+"):
                    coef = 1.0
                elif coef_str == "-":
                    coef = -1.0
                else:
                    coef = float(coef_str)
                coeffs_dict[1] = coeffs_dict.get(1, 0) + coef
            elif power_match:
                # Power term (x^n term)
                coef_str = power_match.group(1)
                if coef_str in ("", "+"):
                    coef = 1.0
                elif coef_str == "-":
                    coef = -1.0
                else:
                    coef = float(coef_str)
                power = int(power_match.group(2))
                coeffs_dict[power] = coeffs_dict.get(power, 0) + coef
            else:
                raise ValueError(f"Invalid term: {term}")
        
        # Create the coefficient array
        if not coeffs_dict:
            return cls([0])
            
        max_degree = max(coeffs_dict.keys())
        coeffs = np.zeros(max_degree + 1)
        
        for degree, coef in coeffs_dict.items():
            coeffs[degree] = coef
            
        return cls(coeffs)