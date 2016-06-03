"""
    Training Script
"""

import tensorflow as tf
import numpy as np
import datetime
import preprocessing
import utils
import scores
from SequentialCNN import SequentialCNN

# Parameters
# ==================================================

#Flags are command-line arguments to our program
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50 , "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes",  "3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 50, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_integer("n_context",2,"Previous and future context.")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 475, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs")
tf.flags.DEFINE_integer("test_every", 100, "Steps to test the trained model with the testing partition")

# Config Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")





# Data Preparatopn
# ==================================================

print("Loading Training data...")
rev_train = preprocessing.load_data('./data/train.xml')
rev_test = preprocessing.load_data('./data/test.xml')
rev_full = utils.get_sentences(rev_train) + utils.get_sentences(rev_test)
vocabulary = utils.build_vocabulary(rev_full)
#print(len(vocabulary))
#print(vocabulary)
#print(sorted(vocabulary))
x_train, y_train = utils.get_formatted_sentences(rev_train,vocabulary,FLAGS.n_context*2+1)
x_test, y_test = utils.get_formatted_sentences(rev_test,vocabulary,FLAGS.n_context*2+1)
#print(x_train)
#print(y_train)
#print("Number of Tweets Train: {:d}".format(x_train.shape[0]))
#print("Vocabulary Size Train: {:d}".format(len(vocabulary)))
#print("Number of Tweets Test: {:d}".format(x_test.shape[0]))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = SequentialCNN(
            sequence_length=FLAGS.n_context*2+1,
            #O-TAG, B-TAG, I-TAG
            num_classes=3,
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            batch_size=FLAGS.batch_size)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #TODO: Experiment with several optimizers
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch,pos):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            
            _, step, loss, accuracy, predictions = sess.run( 
                [train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print(weighted[0])
            #if pos % FLAGS.test_every == 0:
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def test_step(x_test,y_test):
            
            accuracy, predictions = sess.run([cnn.accuracy,cnn.predictions], {cnn.input_x: x_test, cnn.input_y: y_test, cnn.dropout_keep_prob: 1.0})
            #print(predictions[:30])
            #print(predictions)
            return predictions.tolist()


        #Training Phase

        # Generate batches
        batches = utils.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        

        # Training loop. For each batch...
        maxPre = 0
        maxRecall = 0
        maxF1 = 0
        i = 1
        print("(precision recall f1)")
        for batch in batches:
            if len(batch) == FLAGS.batch_size:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch,i)
                current_step = tf.train.global_step(sess, global_step)

                # Test with Testing Partition
                
                if i % FLAGS.test_every == 0:
                    test_batches = utils.batch_iter(
                        list(zip(x_test, y_test)), FLAGS.batch_size, FLAGS.num_epochs)

                    predictions = []
                    for test_batch in test_batches:
                        if len(test_batch) == FLAGS.batch_size:
                            x_test_batch, y_test_batch = zip(*test_batch)
                            predictions += test_step(x_test_batch,y_test_batch)

                    print(predictions[:100])
                    predictions = np.array(predictions)
                    y_test = y_test[:(len(y_test)-len(y_test)%FLAGS.batch_size)]
                    correct_predictions = np.argmax(y_test, axis = 1) 
                    #print(len(correct_predictions))
                    print(correct_predictions[:100])
                    pre,recall,f1 = scores.iobF1(predictions,correct_predictions)
                    print(pre,recall,f1)
                    if f1 > maxF1:
                        maxPre = pre
                        maxRecall = recall
                        maxF1 = f1
                """
                # Test with Testing Partition
                if i % FLAGS.test_every == 0:
                    predictions = test_step(x_test,y_test)
                    correct_predictions = np.argmax(y_test, axis = 1) 
                    pre,recall,f1 = scores.iobF1(predictions,correct_predictions)
                    print(pre,recall,f1)
                    if f1 > maxF1:
                        maxPre = pre
                        maxRecall = recall
                        maxF1 = f1
                """
                i +=1 

        #print("########## TESTING ##########")
        #print(predictions[:100])
        #print(correct_predictions[:100])
        #ok = [0,0,0]
        #i = 0
        #for i in range(len(predictions)):
        #    if predictions[i] == correct_predictions[i]:
        #        ok[predictions[i]] += 1
        

        
        #predicted = [0,0,0]
        #unique, counts = np.unique(predictions, return_counts=True)
        #print(unique)
        #for i in range(len(unique)):
        #    predicted[unique[i]] = counts[i]

        #print("Predicted: ",predicted)

        #total = [0,0,0]
        #ids, idsCounts = np.unique(correct_predictions, return_counts=True)
        #print(idsCounts)
        #for i in range(len(ids)):
        #    total[ids[i]] = idsCounts[i]   
        #print("Total: ",total)

        # Print accuracy
        #print("=======TESTING========")
        #print("Total number of test examples: {}".format(len(y_test)))
        #print("Accuracy: {:g}".format(accuracy)) 

                    
        print("MaxPre",maxPre)
        print("MaxRecall",maxRecall)
        print("MaxF1",maxF1)
        #Testing Phase
        #test_step(x_test,y_test)
