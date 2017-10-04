#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Federal University of Campina Grande (UFCG)
#Author: Ítalo de Pontes Oliveira
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import numpy as np
from numpy import *
import sys

## This method prints the correct way to call this application in terminal
#  @param argv Is the parameter list passed
def printErrorLog(argv):
	message = "\n\n\n"
	message += "Call this applications as:\n\n"
	message += "python " + sys.argv[0] + " <input_file_name> <learning_rate> <stop_criteria> <output_prefix_filename>\n"
	message += "in which:\n"
	message += "<input_file_name>: Is the file name that contains the 'x' and 'y' values separated by the comma to be trained.\n"
	message += "<learning_rate>: Learning rate value (e.g., 0.00003).\n"
	message += "<stop_criteria>: Gradient norma size to stop the training (e.g., 0.000001).\n"
	message += "<lamba value>: Lambda value (e.g., 0.001, 0.01, 0.1).\n"
	message += "<output_prefix_filename>: Graphics filename to save output.\n\n"
	message += "Example:\npython3 " + sys.argv[0] + " sample_treino.csv 0.00003 0.0001 0.001 output"
	message += "\n\n\n"
	print(message)

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
	
def normalize(mat):
	max_values = np.amax( mat, axis = 0 )
	min_values = np.amin( mat, axis = 0 )
	diff = max_values - min_values
	mat_normalized = ( mat - min_values ) / diff
	return [mat_normalized, max_values, min_values]

def compute_cost(H, w, y, lmbda):
	mat = y - np.dot(H, w)
	rss = np.sum( np.transpose( mat ) * mat )
	cost = rss + ( lmbda * np.dot( np.transpose(w), w ) )
	return cost

def compute_norma(vector):
	norma = np.sqrt( np.sum( vector ** 2 ) )
	return norma

def step_gradient(H, w_current, y, learning_rate, lmbda):
	w = 0.0
	norma = 0.0
	N = float(len(H))
	
	partial = np.dot( np.transpose(H), ( y - np.dot( H, w_current ) ) )
	
	w = ( ( 1 - ( 2 * learning_rate * lmbda ) ) * w_current ) + ( 2 * learning_rate * partial )
	
	norma = compute_norma(partial)
	
	return [w, norma, partial]

def gradient_descent(H, y, learning_rate, epsilon, lmbda):
	w = np.zeros((H.shape[1])) #has the same size of output
	cost_total = []
	cost_by_step = 0
	norma_total = []
	norma = epsilon + 1
	num_iterations = 0
	
	while(norma > epsilon):
		[w, norma, partial] = step_gradient(H, w, y, learning_rate, lmbda)
		num_iterations += 1
		if num_iterations % 10000 == 0:
			cost_by_step = compute_cost(H, w, y, lmbda)
			cost_total.append(cost_by_step)
			print("Num iteractions: {0}".format(num_iterations))
			print("Norma: {0}".format(norma))
			print("Coef: {0}".format(w))
			print("Partial: {0}".format(partial))
			print("Cost: {0}\n\n".format(cost_by_step))
		norma_total.append(norma)
	
	return [w, num_iterations, cost_total, norma_total]
 
 

## Main function
if __name__ == '__main__':
	if len(sys.argv) != 6:
		printErrorLog(sys.argv)
		exit(0)

	# Variable declaration	
	input_filename = sys.argv[1]
	learning_rate = float(sys.argv[2])
	epsilon = float(sys.argv[3])
	lmbda = float(sys.argv[4])
	prefix_filename = sys.argv[5]
	
	att = genfromtxt(input_filename, delimiter=",", skip_header=1)
	H = att[:,0:-1] # Get content to be trained
	y = att[:,-1] # Get column of predict variable
	H_with_ones = np.c_[np.ones(len(H)), H]
	
	[w, num_iterations, cost_total, norma_total] = gradient_descent(H_with_ones, y, learning_rate, epsilon, lmbda)
	
	print("\n\nNum iterations: {0}\nCost: {1}\nW: {2}\nNorma: {3}".format(num_iterations, cost_total[-1], w, norma_total[-1]))
	
