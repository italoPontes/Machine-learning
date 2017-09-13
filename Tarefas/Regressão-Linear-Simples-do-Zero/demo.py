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

import numpy
from numpy import *
import sys

## This method prints the correct way to call this application in terminal
#  @param argv Is the parameter list passed
def printErrorLog(argv):
	message = "\n\n\n"
	message += "This application works one of the ways:\n\n"
	message += "python " + sys.argv[0] + " <input_file_name>\n"
	message += "python " + sys.argv[0] + " <input_file_name> <output_figure_name> <learning_rate>\n"
	message += "python " + sys.argv[0] + " <input_file_name> <output_figure_name> <learning_rate> <num_iteractions>\n\n"
	message += "in which:\n"
	message += "<input_file_name>: Is the file name that contains the 'x' and 'y' values separated by the comma to be trained.\n"
	message += "<output_figure_name>: Figure name to save iteraction x RSS graphic.\n"
	message += "<learning_rate>: The rate in which the gradient will be changed in one step.\n"
	message += "<num_iteractions>: Interactions number that the slope line will approximate before a stop.\n"
	message += "\n\n\n"
	print message

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
	plt.close() # closes a window

# y = mx + b
# m is slope, b is y-intercept
## Compute the errors for a given line
#  @param b Is the linear coefficient
#  @param m Is the angular coefficient
#  @param x Domain points
#  @param y Domain points
def compute_error_for_line_given_points(w0, w1, x, y):
	totalError = sum((y - (w1 * x + w0)) ** 2)
	totalError /= float(len(x))
	return totalError

## Calculate a new linear and angular coefficient step by a learning rate. 
#  @param w0_current Current linear coefficient
#  @param w1_current Current linear coefficient
#  @param x Domain points
#  @param y Image points
#  @param learningRate The rate in which the gradient will be changed in one step
def step_gradient(w0_current, w1_current, x, y, learningRate):
	w0_gradient = 0
	w1_gradient = 0
	norma = 0
	N = float(len(x))
	
	w0_gradient = -2 * sum( y - ( w0_current + ( w1_current * x ) ) ) / N
	w1_gradient = -2 * sum( ( y - ( w0_current + ( w1_current * x ) ) ) * x ) / N

	norma = numpy.linalg.norm(w0_gradient - w1_gradient)
	
	new_w0 = w0_current - (learningRate * w0_gradient)
	new_w1 = w1_current - (learningRate * w1_gradient)
	
	return [new_w0, new_w1, norma]

## Run the descending gradient
#  @param x Domain points
#  @param y Image points
#  @param starting_w0 Linear coefficient initial
#  @param starting_w1 Angular coefficient initial
#  @param learning_rate The rate in which the gradient will be changed in one step
#  @param num_iterations Interactions number that the slope line will approximate before a stop.
#  @param output_filename Figure name to save iteration x RSS graphic
def gradient_descent_runner(x, y, starting_w0, starting_w1, learning_rate, num_iterations, output_filename):
	w0 = starting_w0
	w1 = starting_w1
	rss_by_step = 0
	rss_total = []
	norma = learning_rate
	iteration_number = 0
	
	condiction = True
	if num_iterations < 1:
		condiction = False
	
	while (norma > 0.001 and not condiction) or ( iteration_number < num_iterations and condiction):
		w0, w1, norma = step_gradient(w0, w1, x, y, learning_rate)
		
		rss_by_step = compute_error_for_line_given_points(w0, w1, x, y)
		rss_total.append(rss_by_step)
		iteration_number += 1

	save_figure(rss_total, "Iteration", "RSS", output_filename)	
	
	return [w0, w1, iteration_number]

## Compute the W0 and W1 by derivative
#  @param x Domain points
#  @param y Image points
def compute_normal_equation(x, y):
	x_mean = numpy.mean(x)
	y_mean = numpy.mean(y)
	w1 = sum((x - x_mean)*(y - y_mean))/sum((x - x_mean)**2)
	w0 = y_mean-(w1*x_mean)
	return [w0, w1]

## Compute the line approximation for all data passed
#  @param input_filename File that contains the domain and image points
#  @param output_filename Figure name to save iteraction x RSS graphic
#  @param learning_rate The rate in which the gradient will be changed in one step
#  @param num_iteractions Interactions number that the slope line will approximate before a stop
def run(input_filename, output_filename, learning_rate, num_iterations):
	points = genfromtxt(input_filename, delimiter=",")
	initial_w0 = 0 # initial y-intercept guess
	initial_w1 = 0 # initial slope guess
	x = points[:,0] 
	y = points[:,1]
	
	if learning_rate == 0: 
		[w0, w1] = compute_normal_equation(x, y)
	else:
		print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_w0, initial_w1, compute_error_for_line_given_points(initial_w0, initial_w1, x, y))
		[w0, w1, num_iterations] = gradient_descent_runner(x, y, initial_w0, initial_w1, learning_rate, num_iterations, output_filename)
	
	print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, w0, w1, compute_error_for_line_given_points(w0, w1, x, y))

## Main function
#  @param sys.argv[1] Is the file name that contains the 'x' and 'y' values separated by the comma to be trained
#  @param sys.argv[2] Figure name to save iteraction x RSS graphic
#  @param sys.argv[3] The rate in which the gradient will be changed in one step
#  @param sys.argv[4] Interactions number that the slope line will approximate before a stop
if __name__ == '__main__':
	if (len(sys.argv) != 2) and len(sys.argv) != 4 and (len(sys.argv) != 5):
		printErrorLog(sys.argv)
		exit(0)
	
	output_filename = ""
	input_filename = sys.argv[1]
	learning_rate = 0
	num_iteractions = 0
	
	if len(sys.argv) == 5:
		output_filename = sys.argv[2]
		learning_rate = float(sys.argv[3])
		num_iteractions = int(sys.argv[4])
	elif (len(sys.argv) == 4):
		output_filename = sys.argv[2]
		learning_rate = float(sys.argv[3])
	
	
	run(input_filename, output_filename, learning_rate, num_iteractions)
