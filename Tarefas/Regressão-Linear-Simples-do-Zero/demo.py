#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Federal University of Campina Grande (UFCG)
#Author: Ítalo de Pontes Oliveira
#Adapted from: Siraj Raval
#Available at: https://github.com/llSourcell/linear_regression_live

#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

"""This project will calculate linear regression
"""
import matplotlib.pyplot as plt

from numpy import *
import sys

## This method prints the correct way to call this application in terminal
#  @param argv Is the parameter list passed
def printErrorLog(argv):
	message = "\n\n\n"
	message += "You must call:\n"
	message += "python " + sys.argv[0] + " <input_file_name> <output_figure_name> <learning_rate> <num_iteractions>\n"
	message += "in which:\n"
	message += "<input_file_name>: Is the file name that contains the 'x' and 'y' values separated by the comma to be trained.\n"
	message += "<output_figure_name>: Figure name to save iteraction x RSS graphic.\n"
	message += "<learning_rate>: The rate in which the gradient will be changed in one step.\n"
	message += "<num_iteractions>: Interactions number that the slope line will approximate before a stop.\n"
	message += "\n\n\n"
	print message

# y = mx + b
# m is slope, b is y-intercept
## Compute the errors for a given line
#  @param b Is the linear coefficient
#  @param m Is the angular coefficient
#  @param point All points from domain and image
def compute_error_for_line_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b)) ** 2
	return totalError / float(len(points))

## Calculate a new linear and angular coefficient step by a learning rate. 
#  @param b_current Current linear coefficient
#  @param m_current Current linear coefficient
#  @param points All points from domain and image
#  @param learningRate The rate in which the gradient will be changed in one step
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

## Save figure in disk
#  @param data Data to show in the graphic.
#  @param xlabel Text to be shown in abscissa axis.
#  @param ylabel Text to be shown in ordinate axis.
#  @param filename Graphic name
def save_figure(data, xlabel, ylabel, filename):
	plt.plot(data)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(filename)
	plt.cla() # clears an axis
	plt.clf() # clears the entire current figure 
	plt.close() # closes a window,

## Run the descending gradient
#  @param points Domain and image points
#  @param starting_b Linear coefficient initial
#  @param starting_m Angular coefficient initial
#  @param learning_rate The rate in which the gradient will be changed in one step
#  @param num_iterations Interactions number that the slope line will approximate before a stop.
#  @param output_filename Figure name to save iteraction x RSS graphic
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations, output_filename):
	b = starting_b
	m = starting_m
	previous_rss = 0
	rss_by_step = 0
	i = 0
	diff = learning_rate 
	rss_total = []
	
	condiction = True
	if num_iteractions < 1:
		condiction = False
	
	while (diff > (learning_rate/1000) and not condiction) or ( i < num_iteractions and condiction):
		b, m = step_gradient(b, m, array(points), learning_rate)
		
		previous_rss = rss_by_step
		rss_by_step = compute_error_for_line_given_points(b, m, points)
		diff = fabs(rss_by_step - previous_rss)
		
		rss_total.append(rss_by_step)
		i += 1

	save_figure(rss_total, "Iteraction", "RSS", output_filename)	
	
	return [b, m, i]

## Compute the line approximation for all data passed
#  @param input_filename File that contains the domain and image points
#  @param output_filename Figure name to save iteraction x RSS graphic
#  @param learning_rate The rate in which the gradient will be changed in one step
#  @param num_iteractions Interactions number that the slope line will approximate before a stop
def run(input_filename, output_filename, learning_rate, num_iterations):
	points = genfromtxt(input_filename, delimiter=",")
	initial_b = 0 # initial y-intercept guess
	initial_m = 0 # initial slope guess
	
	print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
	print "Running..."
	[b, m, num_iterations] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations, output_filename)
	print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

## Main function
#  @param sys.argv[1] Is the file name that contains the 'x' and 'y' values separated by the comma to be trained
#  @param sys.argv[2] Figure name to save iteraction x RSS graphic
#  @param sys.argv[3] The rate in which the gradient will be changed in one step
#  @param sys.argv[4] Interactions number that the slope line will approximate before a stop
if __name__ == '__main__':
	if (len(sys.argv) != 4) and (len(sys.argv) != 5):
		printErrorLog(sys.argv)
		exit(0)
	
	input_filename = sys.argv[1]
	output_filename = sys.argv[2]
	learning_rate = float(sys.argv[3])
	num_iteractions = 0
	
	if len(sys.argv) == 5:
		num_iteractions = int(sys.argv[4])
	
	run(input_filename, output_filename, learning_rate, num_iteractions)
