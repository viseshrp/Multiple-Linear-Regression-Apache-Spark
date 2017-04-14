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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: linreg <datafile>"
        exit(-1)

    sc = SparkContext(appName="LinearRegression")

    # Input yx file has y_i as the first element of each line
    # and the remaining elements constitute x_i
    yxinputFile = sc.textFile(sys.argv[1])

    yxlines = yxinputFile.map(lambda line: line.split(','))

    def keyA(l):
        l[0] = 1.0
        X = np.asmatrix(np.array(l).astype('float')).T
        return np.dot(X, X.T)


    def keyb(l):
        Y = float(l[0])
        l[0] = 1.0
        X = np.asmatrix(np.array(l).astype('float')).T
        return np.multiply(X.T, Y)


    A = np.asmatrix(yxlines.map(lambda l: ("keyA", keyA(l))).reduceByKey(
        lambda x, y: np.add(x, y)).map(lambda l: l[1]).collect()[0])

    b = np.asmatrix(yxlines.map(lambda l: ("keyb", keyb(l))).reduceByKey(
        lambda x, y: np.add(x, y)).map(lambda l: l[1]).collect()[0])

    beta = np.array(np.dot(np.linalg.inv(A), b)).tolist()
    
    # print the linear regression coefficients in desired output format
    print "beta: "
    for coeff in beta:
        print coeff

    sc.stop()
