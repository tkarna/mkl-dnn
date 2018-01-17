import tensorflow as tf

# NDHWC
input = tf.Variable(tf.random_normal([1,5,5,5,4], seed=1))
# f_d, f_h, f_w, in_c, out_ch
filter = tf.Variable(tf.random_normal([3,3,3,4,2], seed=2)) 

op = tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NDHWC', name="am_conv3d")
init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)

    print("input")
#    print(input.eval())
    print("filter")
#    print(filter.eval())
    print("result")
    result = sess.run(op)
#    print(result)
    print ("Shape of the input: ", (tf.shape(input)).eval()   )
    print ("Shape of the filter: ",  (tf.shape(filter)).eval()   )
    print ("Shape of the result: ", (tf.shape(result)).eval()   )


