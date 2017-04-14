# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
#
# TODO: Write this.
#
# Takes the yx file as input, where on each line y is the first element
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

'''
Viseshprasad Rajendraprasad
vrajend1@uncc.edu
'''

import sys
import numpy as np

from pyspark import SparkContext

# check if an input file exists, else exit.
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: linreg <datafile>"
        exit(-1)

    sc = SparkContext(appName="LinearRegression")

    '''
    Input yx file has y_i as the first element of each line
    and the remaining elements constitute x_i
    This creates a pointer to the input file to be later made an RDD
    when an action is called. RDD is created as a list of strings which
    are each line of the file.
    '''
    yxinputFile = sc.textFile(sys.argv[1])

    '''
    Take each line through a mapper and split at the "," to get
    corresponding dependent and independent variables in each line
    as a list of [yi, xi1, xi2 ...] forming an RDD.
    '''
    yxlines = yxinputFile.map(lambda line: line.split(','))

    '''
    function to calculate and return product of X and X-transpose matrices
    '''
    def keyA(l):
        l[0] = 1.0  # makes the first element zero as defined in X, for beta-0
        # convert input into a float array, then to matrix and do transpose
        X = np.asmatrix(np.array(l).astype('float')).T
        return np.dot(X, X.T)  # product of X and X-transpose matrices

    '''
    function to calculate and return product of X and Y scalar
    '''
    def keyb(l):
        Y = float(l[0])  # dump the first element in Y.
        l[0] = 1.0  # makes the first element zero as defined in X, for beta-0
        # convert input into a float array, then to matrix
        X = np.asmatrix(np.array(l).astype('float')).T
        return np.multiply(X, Y)  # product of X and Y

    # pass the product finding function to mapper which is used for each set of [yi, xi1, xi2 ...]
    # reducer finds sum of all X and X-transpose products
    # which is X'-transpose times X', where X' is a column vector of X-transposes
    A = np.asmatrix(yxlines.map(lambda l: ("keyA", keyA(l))).reduceByKey(
        lambda x, y: np.add(x, y)).map(lambda l: l[1]).collect()[0])

    # pass the product finding function to mapper which is used for each set of [yi, xi1, xi2 ...]
    # reducer finds sum of all X and Y products which is
    # X'-transpose times Y', where X' is a column vector of X-transposes and
    # Y' is a column vector of Y values
    b = np.asmatrix(yxlines.map(lambda l: ("keyb", keyb(l))).reduceByKey(
        lambda x, y: np.add(x, y)).map(lambda l: l[1]).collect()[0])

    # calculate vector of coefficients of hyperplane as product of inverse of matrix A and vector B
    # make it an array and then make it a list for displaying
    beta = np.array(np.dot(np.linalg.inv(A), b)).tolist()

    # print the linear regression coefficients in desired output format
    print('beta: ')
    for coeff in beta:
        print(coeff[0])

    sc.stop()
