import cv2
import numpy
from tqdm import tqdm
from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B

import keras.backend as K

from image_augmentation import random_warp

import tensorflow as tf
print(encoder.summary())

gensims = []
inpsims = []
try:
    encoder  .load_weights( "models/encoder.h5"   )
    decoder_A.load_weights( "models/decoder_A.h5" )
    decoder_B.load_weights( "models/decoder_B.h5" )
except:
    pass

def save_model_weights():
    encoder  .save_weights( "models/encoder.h5"   )
    decoder_A.save_weights( "models/decoder_A.h5" )
    decoder_B.save_weights( "models/decoder_B.h5" )
    print( "save model weights" )

def momentum(m, grads, past_grads):
    grads = grads / tf.reduce_mean(tf.abs(grads), [1, 2, 3], keep_dims=True)
    past_grads = m * past_grads + grads
    return past_grads

images_A = get_image_paths( "data/trump" )
images_B = get_image_paths( "data/cage"  )
images_A = load_images( images_A ) / 255.0
images_B = load_images( images_B ) / 255.0

images_A += images_B.mean( axis=(0,1,2) ) - images_A.mean( axis=(0,1,2) )

print( "press 'q' to stop training and save model" )

batch_size = 14
w1, target_A, w2, w3 = get_training_data( images_A, batch_size )
warped_B, target_B, _, _ = get_training_data( images_B, batch_size )


tga = target_A.copy()


# iter_num = 255
past_grads = numpy.zeros_like(target_A)
m = 0.9
alpha = 0.01
epsilon = 0.2
num_iter = 31
beta_1, beta_2, accum_s, accum_g = tf.cast(0.9,tf.float64),tf.cast(0.999,tf.float64),numpy.zeros_like(target_A),numpy.zeros_like(target_A)


# print(target_A.shape)
# n = []
# n2 = []
# n3 = []
# n4 = []
# for i in target_A:
#   # print(i.shape)
#   n.append(random_warp(i)[0])


# target_A = n
for epoch in tqdm(range(num_iter)):

    i = epoch
    # loss_A = autoencoder_A.train_on_batch( warped_A, target_A )
    # loss_B = autoencoder_B.train_on_batch( warped_B, target_B )
    # print( loss_A, loss_B )

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    augmented = [
      target_A,
      # w1,
      # w2,
      # w3
    ]
    j = 0
    allgrad = numpy.zeros_like(target_A)
    for a in augmented:
        j+=1
        g = K.gradients(encoder.get_output_at(0), encoder.get_input_at(0))    
        out = g[0].eval(session=K.get_session(), feed_dict={encoder.get_input_at(0):a})

        gA = K.gradients(autoencoder_A.output, autoencoder_A.input)    
        outA = gA[0].eval(session=K.get_session(), feed_dict={autoencoder_A.input:a})

        gB = K.gradients(autoencoder_B.output, autoencoder_B.input)
        outB = gB[0].eval(session=K.get_session(), feed_dict={autoencoder_B.input:a})

        allgrad+=outB+outA+out



    grad=numpy.array(allgrad/j)

    grad_normed = grad / tf.reduce_mean(tf.abs(grad), [1,2,3], keep_dims=True)

    accum_g = grad_normed * (1-beta_1) + accum_g * beta_1

    accum_s = tf.multiply(grad_normed,grad_normed) * (1-beta_2) + accum_s * beta_2

    accum_g_hat = tf.divide(accum_g,(1 - tf.pow(beta_1,i+1)))

    accum_s_hat = tf.divide(accum_s,(1 - tf.pow(beta_2,i+1)))

    target_A = target_A + alpha/(tf.sqrt(accum_s_hat)+1e-6)*tf.sign(accum_g_hat)

    # past_grads =  past_grads*m+grads

    # target_A = target_A + alpha*tf.sign(grad)

    # grad = grad / tf.reduce_mean(tf.abs(grad), [1, 2, 3], keep_dims=True)

    # grad = momentum * grad + grad

    # target_A = target_A + alpha * tf.sign(grad)

    target_A = tf.clip_by_value(target_A, tga-epsilon, tga+epsilon)




        # sess.run(tf.global_variables_initializer())
    target_A = K.eval(target_A)

    # assert False


    # if epoch==99 or epoch%5==1 or epoch==1:
    if True:
        print(f'\n\n======{epoch}=======\n')
        # save_model_weights()
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     target_A = target_A.eval()
        # warped_A, target_A = get_training_data( images_A, batch_size )
        # warped_B, target_B = get_training_data( images_B, batch_size )

            
        test_A = target_A[0:14]
        # out = out[0:14]
        test_B = tga[0:14]

        tta = test_A-test_B

        # print(test_B.shape)
        # print(test_A.shape)
        # print(type(test_B))
        # print(type(test_A))
        # print(type(test_B[0]))
        # print(type(test_A[0]))

        figure_A = numpy.stack([
            tta,
            autoencoder_A.predict( test_A ),
            autoencoder_B.predict( test_A ),
            ], axis=1 )
        figure_B = numpy.stack([
            test_A,
            autoencoder_A.predict( test_B ),
            autoencoder_B.predict( test_B ),
            ], axis=1 )

        # print(autoencoder_B.predict( test_A )[0])
        #print(type(autoencoder_B.predict( test_A )[0]))
        gen = autoencoder_B.predict( target_A )
        gen_no = autoencoder_B.predict( tga )
        num = 0
        ssimres = 0
        for g,gn in zip(gen,gen_no):
            num+=1
            ssimres += tf.image.ssim(tf.convert_to_tensor(g), tf.convert_to_tensor(gn), 1)
        s=K.eval(ssimres)/num
        print(s)
        gensims.append(s)
        for g,gn in zip(target_A,tga):
            num+=1
            ssimres += tf.image.ssim(tf.cast(tf.convert_to_tensor(g),tf.float64), tf.cast(tf.convert_to_tensor(gn),tf.float64), 1)
        s=K.eval(ssimres)/num
        print(s)
        inpsims.append(s)

        figure = numpy.concatenate( [ figure_A, figure_B ], axis=0 )
        figure = figure.reshape( (4,7) + figure.shape[1:] )
        figure = stack_images( figure )

        figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')
        print(epoch)
        cv2.imwrite( f'f-img-{str(epoch)}.png', figure )
        # assert False

print('aifgsm')
# print(repr(gensims))
print(gensims)
# print(repr(inpsims))
print(inpsims)


