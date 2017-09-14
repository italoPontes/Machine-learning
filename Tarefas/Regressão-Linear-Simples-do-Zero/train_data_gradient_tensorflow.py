import tensorflow as tf

#w0: -39.4462566791
#w1: 5.59948287412
w0 = tf.Variable([0], dtype=tf.float32)
w1 = tf.Variable([0], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w1 * x + w0

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

learning_rate = float(0.0001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.

x_income = [ 10. , 10.40133779, 10.84280936, 11.24414716, 11.64548495, 12.08695652, 12.48829431, 12.88963211, 13.2909699 , 13.73244147, 14.13377926, 14.53511706, 14.97658863, 15.37792642, 15.77926421, 16.22073579, 16.62207358, 17.02341137, 17.46488294, 17.86622074, 18.26755853, 18.7090301 , 19.11036789, 19.51170569, 19.91304348, 20.35451505, 20.75585284, 21.15719064, 21.59866221, 22. ]

y_income = [ 26.65883878, 27.30643535, 22.13241017, 21.1698405 , 15.19263352, 26.39895104, 17.43530658, 25.50788523, 36.88459469, 39.66610875, 34.39628056, 41.49799354, 44.98157487, 47.03959526, 48.25257829, 57.03425134, 51.49091921, 61.33662055, 57.58198818, 68.55371402, 64.3109253 , 68.95900864, 74.61463928, 71.8671953 , 76.09813538, 75.77521803, 72.48605532, 77.35502057, 72.11879045, 80.2605705 ]

'''
# Traning based on iterations number
iter_number = 20000
for i in range(iter_number):
	sess.run(train, {x: x_income, y: y_income})

print(sess.run(loss, {x: [x_income], y: [y_income]}))
print("w0:", sess.run(w0))
print("w1:", sess.run(w1))
'''

# Training based on gradient size
norma = 1.0
threshold = 0.001

while( norma > threshold ):
	w0_initial, w1_initial = sess.run([w0, w1], {x: x_income, y: y_income})
	sess.run(train, {x: x_income, y: y_income})
	w0_final, w1_final = sess.run([w0, w1], {x: x_income, y: y_income})
	norma_tf = tf.sqrt((w0_final-w0_initial)**2 + (w1_final-w1_initial)**2)
	norma = float(sess.run(norma_tf))

# Training based on closed equation
'''
x_mean = numpy.mean(x)
y_mean = numpy.mean(y)
w1 = sum((x - x_mean)*(y - y_mean))/sum((x - x_mean)**2)
w0 = y_mean-(w1*x_mean)
return [w0, w1]
'''

# evaluate training accuracy
curr_w0, curr_w1, curr_loss = sess.run([w0, w1, loss], {x: x_income, y: y_income})
print("w0: %s w1: %s loss: %s" % (curr_w0, curr_w1, curr_loss))
