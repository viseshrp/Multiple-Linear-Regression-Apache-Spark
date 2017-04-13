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

import sys
import numpy as np

from pyspark import SparkContext


if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength

  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float)

  #
  # Add your code here to compute the array of
  # linear regression coefficients beta.
  # You may also modify the above code.
  #

  

  # print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
      print coeff

  sc.stop()
