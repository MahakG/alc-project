import tensorflow as tf

def cond(i):
	return i < 10
def body(i):
	return tf.add(i,1)

i = tf.constant(0)
size = tf.constant(5)
with tf.Session():
	tf.initialize_all_variables().run()
	result = tf.while_loop(cond,body,[i])
	print(result.eval())