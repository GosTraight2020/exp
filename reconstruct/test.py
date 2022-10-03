import  tensorflow as tf

tf.enable_eager_execution()
tf.executing_eagerly()



with tf.GradientTape() as t:
    b = tf.Variable(1., dtype=tf.float32)
    a = b*b
    g = t.gradient(a, b)
    print(g)