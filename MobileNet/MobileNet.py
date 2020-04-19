import mobilenet_v1
import numpy as np
import os
import argparse
import tensorflow.contrib.slim as slim
import tensorflow as tf
from dataset import Dataset
from tensorflow.examples.tutorials.mnist import input_data
import itertools
from auglib.augmentation import Augmentations, set_seeds
from auglib.dataset_loader import CSVDataset, CSVDatasetWithName
from auglib.meters import AverageMeter
from auglib.test import test_with_augmentation
import torchvision.datasets as datasets
import torch, os
import pickle


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


# from https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data
def tf_confusion_metrics(model, actual_classes, session, feed_dict):
  predictions = tf.argmax(model, 1)
  actuals = tf.argmax(actual_classes, 1)

  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions = tf.ones_like(predictions)
  zeros_like_predictions = tf.zeros_like(predictions)

  tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, ones_like_predictions)
      ),
      "float"
    )
  )

  tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
      ),
      "float"
    )
  )

  fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, ones_like_predictions)
      ),
      "float"
    )
  )

  fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
      ),
      "float"
    )
  )

  tp, tn, fp, fn = \
    session.run(
      [tp_op, tn_op, fp_op, fn_op],
      feed_dict
    )
  return tp, tn, fp, fn



def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data options
    parser.add_argument('--datasource', type=str, default='isic', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./data', help='location to dataset')
    parser.add_argument('--aug_kinds', nargs='+', type=str, default=[], help='augmentations to perform')
    # Model options
    parser.add_argument('--arch', type=str, default='lenet5', help='network architecture to use')
    parser.add_argument('--target_sparsity', type=float, default=0.9, help='level of sparsity to achieve')
    # Train options
    parser.add_argument('--batch_size', type=int, default=10, help='number of examples per mini-batch')
    parser.add_argument('--train_iterations', type=int, default=100, help='number of training iterations')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer of choice')
    parser.add_argument('--lr_decay_type', type=str, default='constant', help='learning rate decay type')
    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--decay_boundaries', nargs='+', type=int, default=[], help='boundaries for piecewise_constant decay')
    parser.add_argument('--decay_values', nargs='+', type=float, default=[], help='values for piecewise_constant decay')
    # Initialization
    parser.add_argument('--initializer_w_bp', type=str, default='vs', help='initializer for w before pruning')
    parser.add_argument('--initializer_b_bp', type=str, default='zeros', help='initializer for b before pruning')
    parser.add_argument('--initializer_w_ap', type=str, default='vs', help='initializer for w after pruning')
    parser.add_argument('--initializer_b_ap', type=str, default='zeros', help='initializer for b after pruning')
    # Logging, saving, options
    parser.add_argument('--logdir', type=str, default='logs', help='location for summaries and checkpoints')
    parser.add_argument('--check_interval', type=int, default=100, help='check interval during training')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval during training')
    args = parser.parse_args()
    # Add more to args
    args.path_summary = os.path.join(args.logdir, 'summary')
    args.path_model = os.path.join(args.logdir, 'model')
    args.path_assess = os.path.join(args.logdir, 'assess')
    return args


args = parse_arguments()

# Dataset
dataset = Dataset(**vars(args))

# 1. Loads data set.
mnist = input_data.read_data_sets('MNIST_data')

# 2. Defines network.
# 2.1 Input feature dim = Nx784, N is sample number
# x = tf.placeholder(tf.float32, [None, 784], name='x-input')
x = tf.placeholder(tf.float32, [10, 224, 224, 3], name='x-input')

# 2.2 Reshapes the feature to Nx28x28 images
# x_image = tf.reshape(x, [-1, 28, 28, 1])
x_image = tf.reshape(x, [10, 224, 224, 3])

# 2.3 Defines the network.
with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):
    y, _ = mobilenet_v1.mobilenet_v1(
        x_image,
        # num_classes=10,
        num_classes=2,
        is_training=True,
    )
# 2.4 Input ground truth labels in on-hot.
# y_ = tf.placeholder(tf.int32, [None, 10], name='y-input')
y_ = tf.placeholder(tf.int32, [None, 2], name='y-input')

# 2.5 Defines Loss function.
loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=y_)

# 2.6 Defines accuracy.
accuracy = tf.metrics.accuracy(labels=tf.argmax(y_, axis=1),
                               predictions=tf.argmax(y, axis=1))

auc = tf.metrics.auc(labels=tf.argmax(y_, axis=1),
                     predictions=tf.argmax(y, axis=1))
global_step = tf.Variable(0, trainable=False)

# 2.7 Train operation is minimizing the loss.
train_operation = tf.train.AdamOptimizer(1e-3)\
    .minimize(loss, global_step=global_step)

# 3. Trains the network
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    # Trains 10000 steps.
    for i in range(10000):
        # Each step uses 100 training samples.
        # xs, ys = mnist.train.next_batch(100)
        batch = dataset.get_next_batch('train', args.batch_size)
        ys = batch['label']
        xs = batch['input']
        # Converts the labels into probability vectors.
        ys_h = get_one_hot(ys, 2)
        # Runs the training operation.
        _, loss_value, accuracy_value, auc_value,  step = \
            sess.run([train_operation, loss, accuracy, auc,  global_step], feed_dict={x: xs, y_: ys_h})

        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        # TP = np.sum(np.logical_and(tf.argmax(y, axis=1) == 1, tf.argmax(y_, axis=1) == 1))
        #
        # # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        # TN = np.sum(np.logical_and(tf.argmax(y, axis=1) == 0, tf.argmax(y_, axis=1) == 0))
        #
        # # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        # FP = np.sum(np.logical_and(tf.argmax(y, axis=1) == 1, tf.argmax(y_, axis=1) == 0))
        #
        # # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        # FN = np.sum(np.logical_and(tf.argmax(y, axis=1) == 0, tf.argmax(y_, axis=1) == 1))

        # tp, tn, fp, fn = tf_confusion_metrics(model, actual_classes, sess, feed_dict={x: xs, y_: ys_h})

        print("After %d training step(s), "
              "on training batch, loss is %g. "
              "accuracy is = %g"
              "AUC is = %g"
              % (step, loss_value, accuracy_value[0], auc_value[0]))
        # print("TP, TN, FP, FN", TP, TN, FP, FN)
        # Saves the model into the disk.
        if i % 1000 == 0:
            saver.save(sess,
                       os.path.join('./mobilenet_v1/', 'model.ckpt'),
                       global_step=global_step)