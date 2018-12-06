
# coding: utf-8

# In[50]:


import sys
sys.path.append('GAN')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import math
import numpy as np
LAMBDA = 10 # Gradient penalty lambda hyperparameter


# In[36]:


# utility function
import scipy
try:
    import moviepy.editor as mpy
except:
    os.system("pip install moviepy")
    import moviepy.editor as mpy

def visualize_imgs(imgs, shape, save_path=None):
    (row, col) = shape[0], shape[1]
    height, width = imgs[0].shape[:2]
    total_img = np.zeros((height*row, width*col))
    for n, img in enumerate(imgs):
        j = int(n/col)
        i = n%col
        total_img[j*height:(j+1)*height,i*width:(i+1)*width] = img
    if save_path is not None:
        scipy.misc.imsave(save_path, img)
    return total_img
  
def make_gif(images, fname, duration=2, true_image=False):    
    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps = len(images) / duration)
def iter_data(*data, **kwargs):
    size = kwargs.get('batch_size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = int(n / size)
    if n % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean",mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(mean-var)))
        tf.summary.scalar("stddev",stddev)
        tf.summary.scalar("max",tf.reduce_max(var))
        tf.summary.scalar("min",tf.reduce_min(var))
        tf.summary.scalar("histogram",var)

# In[2]:


from tensorflow.keras.datasets import mnist
# load data 标准化
(x_train, _),(x_test, _) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)
# In[22]:


class DCGAN(object):
    def __init__(self, image_size, image_channel, z_dim=128, learning_rate=1e-4):
        self.image_size = image_size
        self.image_channel = image_channel
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        #self.build_model()
        
        self.batch_size=100

    def build_model(self):
        if self.image_channel==1:
            self.image_real = tf.placeholder(tf.float32,
                                             [None, self.image_size, self.image_size])
        else:
            self.image_real = tf.placeholder(tf.float32,
                                             [None, self.image_size,
                                              self.image_size, self.image_channel])
            
        # create generator
        self.image_fake = self.generator()
        
        # create discriminator and get its prediction for real/fake image
        self.pred_real, self.logit_real = self.discriminator(self.image_real)
        self.pred_fake, self.logit_fake = self.discriminator(self.image_fake)

        # loss of discriminator
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                          logits=self.logit_real,
                                          labels=tf.ones_like(self.logit_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                          logits=self.logit_fake,
                                          labels=tf.zeros_like(self.logit_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # loss of generator
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                          logits=self.logit_fake,
                                          labels=tf.ones_like(self.logit_fake)))
       
        # create optimize operation for discriminator
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        self.d_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss,
                                                                            var_list=self.d_vars)
        
        # create optimize operation for generator
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        self.g_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss,
                                                                             var_list=self.g_vars)

    def discriminator(self, image):
        lrelu = tf.nn.leaky_relu
        conv2d = tf.layers.conv2d
        bn = tf.layers.batch_normalization
        linear = tf.layers.dense    
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            if self.image_channel==1:
                image = tf.reshape(image, [-1, self.image_size, self.image_size, 1])
            hidden = self.z
            hidden = image
            hidden = lrelu(conv2d(hidden, 32, kernel_size=5, strides=2, padding='same'))
            hidden = lrelu(bn(conv2d(hidden, 128, kernel_size=5, strides=2, padding='same'),
                              training=True))
            hidden = tf.layers.flatten(hidden)
            hidden = lrelu(bn(linear(hidden, 1024), training=True))
            hidden = linear(hidden, 1)
            tf.summary.histogram("D_output",tf.nn.sigmoid(hidden))
            return tf.nn.sigmoid(hidden), hidden

    def generator(self, y=None):
        relu = tf.nn.relu
        deconv2d = tf.layers.conv2d_transpose
        bn = tf.layers.batch_normalization
        linear = tf.layers.dense
        with tf.variable_scope("generator"):
            self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
            hidden = self.z
            hidden = relu(bn(linear(hidden, 1024), training=True))
            hidden = relu(bn(linear(hidden, (self.image_size//4)*(self.image_size//4)*128),
                             training=True))
            hidden = tf.reshape(hidden, [-1, self.image_size//4, self.image_size//4, 128])
            hidden = relu(bn(deconv2d(hidden, 32, kernel_size=5, strides=2, padding='same'),
                             training=True))
            hidden = tf.nn.sigmoid(deconv2d(hidden, self.image_channel, kernel_size=5,
                                            strides=2, padding='same'))
            if self.image_channel==1:
                hidden = tf.reshape(hidden, [-1, self.image_size, self.image_size])
            tf.summary.histogram("G_output",hidden)
            return hidden
          
    def train(self, sess, x_train, num_epoch=100, batch_size=100, num_sample=100,
              show_samples=True, sample_path='./samples', n_critic=2, log=False):
        # sample some random noise, these noise is used to monitor generated image 
        sample_z = np.random.uniform(-1, 1, size=(num_sample , self.z_dim))
        sample_imgs = []
        
        counter = 1
        start_time = time.time()
        d_loss_epoch = []
        g_loss_epoch = []
        for epoch in range(num_epoch):
            shuffle_idx = np.random.permutation(len(x_train))
            x_train = x_train[shuffle_idx]
            d_losses = []
            g_losses = []
            for batch_images in iter_data(x_train, batch_size=batch_size):
                batch_z = np.random.uniform(-1, 1,
                                            [batch_size, self.z_dim]).astype(np.float32)
                if counter % (n_critic+1) != 0:
                    # Update D network
                    feed_dict={ 
                        self.image_real: batch_images,
                        self.z: batch_z,
                    }
                    d_loss, _ = sess.run([self.d_loss, self.d_update_op],
                                         feed_dict=feed_dict)
                    d_losses.append(d_loss)
                else:
                    # Update G network
                    g_loss, _ = sess.run([self.g_loss, self.g_update_op],
                                         feed_dict={self.z: batch_z})
                    g_losses.append(g_loss)
                counter += 1
            if log:
                print("Epoch: [{}] time: {:.2f}, d_loss: {:.4f}, g_loss: {:.4f}".format(
                      epoch, time.time()-start_time, np.mean(d_losses), np.mean(g_losses)))
            d_loss_epoch.append(np.mean(d_losses))
            g_loss_epoch.append(np.mean(g_losses))
            
            # save generated samples
            samples = sess.run(self.image_fake, feed_dict={self.z: sample_z})
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            img = visualize_imgs(samples, shape=(10,20),
                                 save_path=sample_path+'/epoch-{}.jpg'.format(epoch))
            sample_imgs.append(img)
                
            if (epoch+1) % 10 == 0:
                if show_samples:
                    plt.imshow(img, cmap = 'gray')
                    plt.axis('off')
                    plt.title('epoch {}'.format(epoch+1))
                    plt.show()
        return sample_imgs, d_loss_epoch, g_loss_epoch
      
    def save_model(self, sess, checkpoint_dir='./checkpoints', model_name='model', step=None):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()
        if step is not None:
            saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        else:
            saver.save(sess, os.path.join(checkpoint_dir, model_name))
        
    def load_model(self, sess, checkpoint_dir='./checkpoints', model_name='model', step=None):
        saver = tf.train.Saver()
        if step is not None:
            saver.save(sess, os.path.join(checkpoint_dir, model_name+'-{}'.format(step)))
        else:
            saver.save(sess, os.path.join(checkpoint_dir, model_name))
        


# In[58]:


#Inherit from DCGAN class
class WGAN_GP(DCGAN):
    
    def build_model(self):
        
        batch_size=100
        
        if self.image_channel==1:
            self.image_real = tf.placeholder(tf.float32,
                                             [None, self.image_size, self.image_size])
        else:
            self.image_real = tf.placeholder(tf.float32,
                                             [None, self.image_size,
                                              self.image_size, self.image_channel])
        sess = tf.InteractiveSession()
        

    
        # create generator
        self.image_fake = self.generator()
        
        # create discriminator and get its prediction for real/fake image
        self.pred_real, self.logit_real = self.discriminator(self.image_real)#return tf.nn.sigmoid(hidden), hidden
        self.pred_fake, self.logit_fake = self.discriminator(self.image_fake)#return tf.nn.sigmoid(hidden), hidden

        # loss of discriminator
        self.d_loss_real = tf.reduce_mean(self.logit_real)
        self.d_loss_fake = tf.reduce_mean(self.logit_fake)                        
        self.d_loss = self.d_loss_fake - self.d_loss_real
        
        """
        ==================================================================
        real_X = tf.placeholder(tf.float32, shape=[batch_size, mnist_dim])
        random_X = tf.placeholder(tf.float32, shape=[batch_size, random_dim])
        random_Y = G(random_X)  == self.image_fake
        eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
        X_inter = eps*real_X + (1. - eps)*random_Y
        grad = tf.gradients(D(X_inter), [X_inter])[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))
        """
        alpha = tf.random_uniform(shape=[batch_size,self.image_size,self.image_size], 
        minval=0.,
        maxval=1.)
        differences = self.image_fake - self.image_real

        interpolates = self.image_real + (alpha*differences)

        gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

        # 浪打项
        gradient_penalty = tf.reduce_mean(tf.square(slopes-1.))

        #L(i) 项
        self.d_loss += LAMBDA*gradient_penalty 
        tf.summary.histogram("D_loss",self.d_loss)
      
        # loss of generator
        self.g_loss = -tf.reduce_mean(self.logit_fake)
        tf.summary.histogram("G_loss",self.g_loss)         
        # create optimize operation for discriminator
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
       
        self.d_update_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.d_loss,
                                                                        var_list=self.d_vars)
        
        # create optimize operation for generator
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        self.g_update_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss,
                                                                        var_list=self.g_vars)
        

        #tf.summary.image("G_output_image", image)
        

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("log/train",sess.graph)
        self.test_writer = tf.summary.FileWriter("log/test",sess.graph)
        
        sess.close()
        # create weight cliping operation
        self.weight_clip_ops = []
#         for var in self.d_vars:            
#             self.weight_clip_ops.append(var.assign(tf.clip_by_value(var, -0.01, 0.01)))
            
    def train(self, sess, x_train, num_epoch=100, batch_size=100, 
              num_sample=100, show_samples=True, n_critic=2, 
              sample_path='./samples', log=False): 
    
        # sample some random noise, these noise is used to monitor generated image 
        sample_z = np.random.uniform(-1, 1, size=(num_sample , self.z_dim))
        sample_imgs = []
        
        saver = tf.train.Saver()
        
        counter = 1
        start_time = time.time()        
        d_loss_epoch = []
        g_loss_epoch = []
        for epoch in range(num_epoch):
            shuffle_idx = np.random.permutation(len(x_train))
            x_train = x_train[shuffle_idx]
            d_losses = []
            g_losses = []
            for batch_images in iter_data(x_train, batch_size=batch_size):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                batch_z = np.random.uniform(-1, 1,
                                            [batch_size, self.z_dim]).astype(np.float32)
                if counter % (n_critic+1) != 0:
                    # Update D network
                    feed_dict={ 
                        self.image_real: batch_images,
                        self.z: batch_z,
                    }
                    d_loss, _ ,summary= sess.run([self.d_loss, self.d_update_op,self.merged],
                                         feed_dict=feed_dict,options=run_options,run_metadata=run_metadata)
                    


                    self.train_writer.add_run_metadata(run_metadata,"iteration %d"%counter)
                    self.train_writer.add_summary(summary,counter)
                    saver.save(sess,"models/model.ckpt",counter)
                    
                    d_losses.append(d_loss)
                    sess.run(self.weight_clip_ops)
                else:
                    # Update G network
                    g_loss, _ = sess.run([self.g_loss, self.g_update_op],
                                         feed_dict={self.z: batch_z})
                    g_losses.append(g_loss)
                counter += 1                                        
            if log:
                print("Epoch: [{}] time: {:.2f}, d_loss: {:.4f}, g_loss: {:.4f}".format(
                      epoch, time.time()-start_time, np.mean(d_losses), np.mean(g_losses)))
            d_loss_epoch.append(np.mean(d_losses))
            g_loss_epoch.append(np.mean(g_losses))
            
            # save generated samples
            samples = sess.run(self.image_fake, feed_dict={self.z: sample_z})

            self.image = make_image(samples)

            #tf.summary.image("image_input",self.image,10)

            imgsummary = tf.summary(value=[tf.summary.Value("op fake image", image=self.image)])
            writer.add_summary(imgsummary, epoch)

            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            img = visualize_imgs(samples, shape=(10,20),
                                 save_path=sample_path+'/epoch-{}.jpg'.format(epoch))
            
            
        
            sample_imgs.append(img)
                
            if (epoch+1) % 10 == 0:
                self.save_model(sess, step=counter)
                if show_samples:
                    plt.imshow(img, cmap = 'gray')
                    plt.axis('off')
                    plt.title('epoch {}'.format(epoch+1))
                    plt.show()

        self.train_writer.close()
        self.test_writer.close()            
        return sample_imgs, d_loss_epoch, g_loss_epoch
      


# In[59]:


tf.reset_default_graph()
tf.set_random_seed(123)
np.random.seed(123)
wgan_gp = WGAN_GP(image_size=28, image_channel=1, learning_rate=5e-5)
wgan_gp.build_model()

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[60]:


sample_z = np.random.uniform(-1, 1, size=(200 , wgan_gp.z_dim))
samples = sess.run(wgan_gp.image_fake, feed_dict = {wgan_gp.z: sample_z})

plt.imshow(samples[0].reshape(28,28), cmap='gray')
plt.axis('off')
plt.title('Generated sample')
plt.show()

samples = x_train[:200]
img = visualize_imgs(samples, shape=(10,20))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Real MNIST samples')
plt.show()


# In[61]:


# Start training
sample_imgs, d_loss_epoch, g_loss_epoch = wgan_gp.train(sess, x_train, num_sample=200, num_epoch=50, n_critic=5)


# In[62]:


imgs = np.array(sample_imgs)
make_gif(imgs*255., 'GAN/wgan.gif', true_image=True, duration=2)

from IPython.display import Image
Image(url='GAN/wgan.gif')  

