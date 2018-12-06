
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_data", one_hot=True)


# In[2]:


# input shape == [28,28]  784
# conv1
# maxpool
# conv2
# maxpool
# fc1
# fc2
# [10]


# In[3]:


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape)) # shape = shape!

def con2d(x,W):
    return tf.nn.conv2d(x,W,[1,1,1,1],padding="SAME")

def max_pool_2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


# In[4]:



ipx = tf.placeholder(tf.float32,shape=[None,784])
opy = tf.placeholder(tf.float32,shape=[None,10])

x_image = tf.reshape(ipx,[-1,28,28,1])
#conv1 

w1 = weight([5,5,1,32])
b1 = bias([32])  # 32 lay each bias


conv1 = tf.nn.relu(con2d(x_image,w1)+b1)
pool1 = max_pool_2d(conv1)


w2 = weight([5,5,32,64])
b2 = bias([64])
conv2 = tf.nn.relu(con2d(pool1,w2)+b2)
pool2 = max_pool_2d(conv2)
pool2_reshape = tf.reshape(pool2,[-1,3136]) # n_sample = ?  7*7*64


fcw1 = weight([3136,1024]) #(28/2/2)
fcb1 = bias([1024])
fc1 = tf.nn.relu(tf.matmul(pool2_reshape,fcw1)+fcb1)
fc1 = tf.nn.dropout(fc1,keep_prob=0.8)

fcw2 = weight([1024,10])
fcb2 = bias([10])
fc2 = tf.nn.relu(tf.matmul(fc1,fcw2)+fcb2)
result = tf.nn.softmax(fc2)


# optimazer

# In[5]:



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=result,
    labels =opy 
))

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# show acc

# In[6]:


correct_prediction = tf.equal(
      tf.argmax(result,1),
      tf.argmax(opy,1)   )

acc = tf.reduce_mean(tf.cast(correct_prediction,"float"))# cast 转 float 用


# In[7]:


from tqdm import tqdm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(5):
        
        for i in tqdm(range(1000)):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step,feed_dict={ipx:batch_xs,opy:batch_ys})
            
        print(sess.run(acc))

