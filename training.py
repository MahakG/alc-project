"""
    Training Script
"""

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import preprocessing
from SequentialCNN import SequentialCNN

# Parameters
# ==================================================

#Flags are command-line arguments to our program
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_integer("n_context", 2,"Previous and future context.")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs")

# Config Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")




# Load data
print("Loading Training data...")
reviews = preprocessing.load_data('./data/train.xml')
vocabulary = utils.getVocabulary(reviews)
print(reviews)

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = SequentialCNN(
            sequence_length=(FLAGS.n_context*2+1),
            #Labeling IOB2 scheme 
            #O, B-, I-
            num_classes=3,
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #TODO: Experiment with several optimizers
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        #TODO: Add Summaries to the training process

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        # Generate batches
        batches = utils.batch_iter(
            list(zip(x, y)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))