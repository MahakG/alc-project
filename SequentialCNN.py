"""
    A CNN Model for Sentence-Level Sentiment Classification
"""

import tensorflow as tf
import utils


class SequentialCNN(object):
    
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, batch_size):
    
        # Variables to feed into the network
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Performs the word2vec representation using our data
        # TODO: Add the possibility to use a pre-trained Word2Vec

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            #self.shape = tf.shape(self.embedded_chars)
            """
            size = batch_size * sequence_length
            middle = sequence_length/2
            indices = []
            while middle < size:
                for i in range(sequence_length):
                    indices.append(middle)
                middle += sequence_length
                

            b = tf.reshape(self.embedded_chars,[size,embedding_size])
            w = tf.gather(b,indices)

            b = tf.to_float(b)
            w = tf.to_float(w)
            
            #HardCoded for sequenceLength 3, needs to be generated
            weight = tf.Variable([0.1, 0.2, 1, 0.2, 0.1])
            invWeight = tf.Variable([0.9, 0.8, 0, 0.8, 0.9])
            weight = tf.tile(weight,[batch_size])
            invWeight = tf.tile(invWeight,[batch_size])

            xW = tf.transpose(tf.mul(invWeight,tf.transpose(b)))
            wW = tf.transpose(tf.mul(weight,tf.transpose(w)))
            
            self.weighted_embedded_chars = tf.reshape(tf.add(xW,wW),[batch_size,sequence_length,embedding_size])
            """
            
            
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    #TODO: Understand difference between Narrow vs Wide Convolution 
                    padding="VALID",
                    name="conv")
                # Apply RELU
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # 1-Maxpool over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

"""
    def cond(self,i):
        return self.i < self.size

    def body(self,i):

        x = tf.squeeze(tf.slice(self.embedded_chars,[1,0,0],[1,self.sequence_length, self.embedding_size]))
        
        midWord = tf.slice(x,[self.sequence_length/2+1,0],[1,self.embedding_size])
        midWord = tf.squeeze(midWord)
        midWordMatrix = tf.tile(midWord,[self.sequence_length])
        midWordMatrix = tf.reshape(midWordMatrix,[self.sequence_length, self.embedding_size])
        
        #HardCoded for sequenceLength 5, needs to be generated
        weight = tf.Variable([0.2,0.5,1,0.5,0.2])
        invWeight = tf.Variable([0.8,0.5,0,0.5,0.8])
        
        x = tf.transpose(x)
        w = tf.transpose(midWordMatrix)

        wX = tf.mul(x,invWeight) 
        wW = tf.mul(w,weight)

        result = tf.add(wX,wW)
        result = tf.transpose(result)
        result = tf.reshape(result,[1,self.sequence_length,self.embedding_size])

        self.weighted_embedded_chars = tf.concat(0,[self.weighted_embedded_chars,result])
        self.i +=1
        
        return [i]
"""   