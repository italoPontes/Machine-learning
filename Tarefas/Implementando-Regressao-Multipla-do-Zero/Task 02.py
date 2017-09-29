#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Federal University of Campina Grande (UFCG)
#Author: Ítalo de Pontes Oliveira
#Adapted from: Siraj Raval
#Available at: https://github.com/llSourcell/linear_regression_live

"""This project will calculate Multiple Linear Regression
"""
import matplotlib.pyplot as plt

import numpy as np
from numpy import *
import sys

## This method prints the correct way to call this application in terminal
#  @param argv Is the parameter list passed
def printErrorLog(argv):
	message = "\n\n\n"
	message += "This application works one of the ways:\n\n"
	message += "python " + sys.argv[0] + " <input_file_name>\n"
	message += "python " + sys.argv[0] + " <input_file_name> <output_figure_name> <learning_rate>\n"
	message += "python " + sys.argv[0] + " <input_file_name> <output_figure_name> <learning_rate> <num_iterations>\n\n"
	message += "in which:\n"
	message += "<input_file_name>: Is the file name that contains the 'x' and 'y' values separated by the comma to be trained.\n"
	message += "<output_figure_name>: Figure name to save iteraction x RSS graphic.\n"
	message += "<learning_rate>: The rate in which the gradient will be changed in one step.\n"
	message += "<num_iterations>: Interactions number that the slope line will approximate before a stop.\n"
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
	

def compute_RSS(H, w, y):
	mat = y - np.dot(H, w)
	rss = np.sum( np.transpose( mat ) * mat ) 
	return rss

def compute_norma(vector):
	norma = np.sqrt( np.sum( vector ** 2 ) )
	return norma

def step_gradient(H, w_current, y, learning_rate):
	w = 0.0
	norma = 0.0
	N = float(len(H))
	
	partial = np.sum( np.transpose(H) * ( y - np.dot(H, w_current) ), axis = 1 )
	
	norma = compute_norma(partial)
	
	w = w_current + ( 2 * learning_rate * partial )
	
	return [w, norma]

def gradient_descent(H, y, learning_rate, epsilon):
	w = np.zeros((H.shape[1])) #has the same size of output
	rss_total = []
	rss_by_step = 0
	norma = epsilon+1
	num_iterations = 0
	
	while(norma > epsilon):
		[w, norma] = step_gradient(H, w, y, learning_rate)
		num_iterations += 1
		if num_iterations % 10 == 0:
			rss_by_step = compute_RSS(H, w, y)
			rss_total.append(rss_by_step)
	
	return [w, num_iterations, rss_total]
 
 

## Main function
if __name__ == '__main__':
	'''
	if (len(sys.argv) != 2) and len(sys.argv) != 4 and (len(sys.argv) != 5):
		printErrorLog(sys.argv)
		exit(0)
	'''
	# Variable declaration	
	input_filename = sys.argv[1]
	learning_rate = float(sys.argv[2])
	epsilon = float(sys.argv[3])
	output_filename = sys.argv[4]
	
	att = genfromtxt(input_filename, delimiter=",", skip_header=1)
	H = att[:,0:-1]
	y = att[:,-1]
	
	H = np.c_[np.ones(len(H)), H]
	
	[w, num_iterations, rss_total] = gradient_descent(H, y, learning_rate, epsilon)
	
	print("Num iterations: {0}\nCoefficients: {1}".format(num_iterations, w))
	
	save_figure(rss_total, "Iteration", "RSS", output_filename)
