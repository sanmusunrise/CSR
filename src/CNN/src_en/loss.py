import tensorflow as tf
from tfnlp.embedding.GloveEmbeddings import *
from tfnlp.layer.Conv1DLayer import *
from tfnlp.layer.NegativeMaskLayer import *
from tfnlp.layer.MaskLayer import *
from tfnlp.layer.DenseLayer import *
from tfnlp.layer.MaskedSoftmaxLayer import *


def weight_loss(logits,label_ids,positive_idx = None,negative_idx = None,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    class_weight = [1.0] * label_size
    class_weight[0] = 0.2
    class_weight = tf.Variable(class_weight,dtype = tf.float32,name = "cls_weight",trainable = False)
    
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    loss_weight = tf.gather(class_weight,label_ids)
    
    loss = - loss_weight * tf.log(golden_prob + 1e-8)
    return loss
    
def f1_reweight_loss(logits,label_ids,positive_idx,negative_idx,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    m = tf.reduce_sum(positive_idx)
    n = tf.reduce_sum(negative_idx)
    p1 = tf.reduce_sum(positive_idx * golden_prob)
    p2 = tf.reduce_sum(negative_idx * golden_prob)
    neg_weight = p1 / (m+n-p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    loss_weight = all_one * positive_idx + all_one * neg_weight * negative_idx
    
    loss = - loss_weight * tf.log(golden_prob +1e-8)
    return loss

def f1_reweight_stopped(logits,label_ids,positive_idx,negative_idx,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    m = tf.reduce_sum(positive_idx)
    n = tf.reduce_sum(negative_idx)
    p1 = tf.reduce_sum(positive_idx * golden_prob)
    p2 = tf.reduce_sum(negative_idx * golden_prob)
    neg_weight = p1 / (m+n-p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    loss_weight = all_one * positive_idx + all_one * neg_weight * negative_idx
    loss_weight_stopped = tf.stop_gradient(loss_weight)
    loss = - loss_weight_stopped * tf.log(golden_prob +1e-8)
    return loss
    
'''
#Use f1_reweight_stooped instead.

def f1_reweight_static_loss(logits,label_ids,is_negative):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    m = tf.reduce_sum(is_negative)
    n = tf.reduce_sum(1- is_negative)
    p1 = tf.reduce_sum(is_negative * golden_prob)
    p2 = tf.reduce_sum((1-is_negative) * golden_prob)
    neg_weight = p1 / (m+n-p2 + 1e-8)
    
    all_one = tf.ones(tf.shape(golden_prob))
    loss_weight = all_one * is_negative + all_one * neg_weight * (1-is_negative)
    loss = - tf.reduce_mean( loss_weight * tf.log(golden_prob +1e-8))
    
    total_instance = tf.cast(tf.shape(golden_prob),dtype = tf.float32)
    positive_loss = -tf.reduce_sum(is_negative * tf.log(golden_prob +1e-8)) / total_instance
    negative_loss = -tf.reduce_sum((1-is_negative) * tf.log(golden_prob +1e-8)) / total_instance

    all_one = tf.ones(tf.shape(golden_prob))
    loss_weight = all_one * is_negative + all_one * neg_weight * (1-is_negative)
    loss = - tf.reduce_mean( loss_weight * tf.log(golden_prob +1e-8))
    
    total_instance = tf.cast(tf.shape(golden_prob),dtype = tf.float32)
    positive_loss = -tf.reduce_sum(is_negative * tf.log(golden_prob +1e-8)) / total_instance
    negative_loss = -tf.reduce_sum((1-is_negative) * tf.log(golden_prob +1e-8)) / total_instance
    #negative_loss = -tf.reduce_sum((1-is_negative) * tf.log(golden_prob +1e-8)) * neg_weight / total_instance

    trainer = tf.train.AdadeltaOptimizer(learning_rate=1,epsilon=1e-06)
    
    positive_grads = trainer.compute_gradients(positive_loss)
    negative_grads = trainer.compute_gradients(negative_loss)
    var_to_grad = {}
    for grad,var in positive_grads:
        var_to_grad[var] = tf.convert_to_tensor(grad)
    for grad,var in negative_grads:
        var_to_grad[var] += tf.convert_to_tensor(grad) * neg_weight
    grads_and_vars = []
    for var in var_to_grad:
        grads_and_vars.append((var_to_grad[var],var))

    train_step = trainer.apply_gradients(grads_and_vars)
    
    return train_step
'''

def likelihood_loss(logits,label_ids,positive_idx = None,negative_idx = None,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    loss = - tf.log(golden_prob +1e-8)
    return loss

def focal_loss(logits,label_ids,positive_idx = None,negative_idx = None,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    
    loss_weight = tf.pow(1.03 - golden_prob,2)
    loss = - loss_weight * tf.log(golden_prob + 1e-8)
    return loss

'''
def confusion_loss(logits,label_ids,is_negative,correct_class_weight,wrong_confusion_matrix):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    log_probs = tf.log(probs + 1e-8)
    log_one_minus_prob = tf.log(1-probs + 1e-8)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    golden_log_prob = tf.gather_nd(log_probs,label_with_idx)

    positvie_confusion_weight = tf.gather(correct_class_weight,label_ids)
    negative_confusion_weight = tf.gather(wrong_confusion_matrix,label_ids)   #B*label_size
    positive_cost = positvie_confusion_weight * golden_log_prob
    negative_cost = tf.reduce_sum(negative_confusion_weight  * log_one_minus_prob,axis = 1)
    cost = positive_cost + negative_cost

    m = tf.reduce_sum(is_negative)
    n = tf.reduce_sum(1- is_negative)
    p1 = tf.reduce_sum(is_negative * golden_prob)
    p2 = tf.reduce_sum((1-is_negative) * golden_prob)
    neg_weight = p1 / (m+n-p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    loss = - cost
    return train_step
''' 

def f1_confusion_loss(logits,label_ids,positive_idx,negative_idx,correct_class_weight,wrong_confusion_matrix, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    log_probs = tf.log(probs + 1e-8)
    log_one_minus_prob = tf.log(1-probs + 1e-8)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    golden_log_prob = tf.gather_nd(log_probs,label_with_idx)

    positvie_confusion_weight = tf.gather(correct_class_weight,label_ids)
    negative_confusion_weight = tf.gather(wrong_confusion_matrix,label_ids)   #B*label_size
    positive_cost = positvie_confusion_weight * golden_log_prob
    negative_cost = tf.reduce_sum(negative_confusion_weight  * log_one_minus_prob,axis = 1)
    cost = positive_cost + negative_cost

    m = tf.reduce_sum(positive_idx)
    n = tf.reduce_sum(negative_idx)
    p1 = tf.reduce_sum(positive_idx * golden_prob)
    p2 = tf.reduce_sum(negative_idx * golden_prob)
    neg_weight = p1 / (m+n-p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    balanced_weight = all_one * positive_idx + all_one * neg_weight * negative_idx
    loss = - balanced_weight * cost
    return loss

def f1_entropy_loss(logits,label_ids,positive_idx = None,negative_idx = None,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    wrong_confusion_matrix = 1.0- tf.diag(tf.Variable([1.0] * label_size,dtype = tf.float32,trainable = False))
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    log_probs = tf.log(probs + 1e-8)
    negative_entropy = probs * tf.log(probs + 1e-8)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    golden_log_prob = tf.gather_nd(log_probs,label_with_idx)

    negative_confusion_weight = tf.gather(wrong_confusion_matrix,label_ids)   #B*label_size
    positive_cost = golden_log_prob
    negative_cost = tf.reduce_sum(- negative_confusion_weight  * negative_entropy,axis = 1)
    cost = positive_cost + negative_cost

    m = tf.reduce_sum(positive_idx)
    n = tf.reduce_sum(negative_idx)
    p1 = tf.reduce_sum(positive_idx * golden_prob)
    p2 = tf.reduce_sum(negative_idx * golden_prob)
    neg_weight = p1 / (m+n-p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    balanced_weight = all_one * positive_idx + all_one * neg_weight * negative_idx
    loss = - balanced_weight * cost
    return loss
