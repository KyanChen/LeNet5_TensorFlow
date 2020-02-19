import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import inference
import train

EVAL_INTERVAL_SECS = 10
BATCH_SIZE = 1000

import time
import numpy as np

def evaluate(mnist):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE,
                                              inference.IMAGE_SIZE,
                                              inference.IMAGE_SIZE,
                                              inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, inference.OUTPUT_NODE], name='y-input')

        xs, ys = mnist.validation.next_batch(BATCH_SIZE)

        print(xs.shape)
        print(ys.shape)

        reshape_xs = np.reshape(xs, (-1, inference.IMAGE_SIZE,
                                     inference.IMAGE_SIZE,
                                     inference.NUM_CHANNELS))
        print(reshape_xs.shape)
        print(mnist.validation.labels[0])

        val_feed = {x: reshape_xs, y_: ys}

        y = inference.inference(x, False, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)

        val_to_restore = variable_average.variables_to_restore()

        saver = tf.train.Saver(val_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=val_feed)
                    print('After %s train ,the accuracy is %g' % (global_step, accuracy_score))
                else:
                    print('No Checkpoint file find')
                    # continue
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("..\Study\MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()



