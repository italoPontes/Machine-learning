"""This project will calculate linear regression
"""

#Federal University of Campina Grande (UFCG)
#Author: Ítalo de Pontes Oliveira
#Adapted from: Siraj Raval
#Available at: https://github.com/llSourcell/linear_regression_live

#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *
import sys


## This method prints the correct way to call this application in terminal
#  @param argv Is the parameter list passed
def printErrorLog(argv):
	message = "\n\n\n"
	message += "You must call:\n"
	message += "python " + sys.argv[0] + " <input_file_name>\n"
	message += "in which:\n"
	message += "<input_file_name>: is the file name that contains the 'x' and 'y' values separated by the comma to be trained."
	message += "\n\n\n"
	print message

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b)) ** 2
	return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)
	return [b, m]

def run(input_filename):
	points = genfromtxt(input_filename, delimiter=",")
	learning_rate = 0.0001
	initial_b = 0 # initial y-intercept guess
	initial_m = 0 # initial slope guess
	num_iterations = 1000
	print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
	print "Running..."
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))



if __name__ == '__main__':
	if len(sys.argv) != 3:
		printErrorLog(sys.argv)
		exit(0)
	
	input_filename = sys.argv[1]
	run(input_filename)
