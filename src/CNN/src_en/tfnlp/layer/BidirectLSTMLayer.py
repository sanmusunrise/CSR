import tensorflow as tf
from Layer import Layer
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import *
from tensorflow.python.util import nest

class BidirectLSTMLayer(Layer):

    def __call__(self,inputs,seq_len):
        if self.output_dim % 2 !=0:
            print "The output dimension of BidirectLSTMLayer should be even. "
            exit(-1)
            
        with tf.variable_scope(self.scope) as scope:
            self.check_reuse(scope)
            #scope.reuse_variables()
            f_cell = LSTMCell(self.output_dim /2 ,initializer = self.initializer(dtype = inputs.dtype))
            b_cell = LSTMCell(self.output_dim /2 ,initializer = self.initializer(dtype = inputs.dtype))
            #rnn.bidirectional_dynamic_rnn(cell,cell,inputs,seq_len,dtype = inputs.dtype)
            return rnn.bidirectional_dynamic_rnn(f_cell,b_cell,inputs,seq_len,dtype = inputs.dtype)

if __name__ =="__main__":

    a = tf.Variable([[[1.0,2],[3,4]],[[5,6],[7,8]]])
    lstm = BidirectLSTMLayer("BiLSTM",4)
    sess = tf.Session()
    output = lstm(a,seq_len = [1,2])  
    vec = tf.concat([output[0][0],output[0][1]],axis = 2)
    sess.run(tf.global_variables_initializer())
    print sess.run(vec)
    '''
    output = lstm(a,seq_len = [1,2])
    print tf.trainable_variables()
    output2 = lstm(a,[2,2])
    print tf.trainable_variables()
    '''
