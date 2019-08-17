import tensorflow as tf
import numpy as np
"""conv1d_tranpose function"""
def conv1d_transpose_wrap(value,
                          filter,
                          output_shape,
                          stride,
                          padding="SAME",
                          data_format="NWC",
                          name=None):
    """Wrap the built-in (contrib) conv1d_transpose function so that output
    has a batch size determined at runtime, rather than being fixed by whatever
    batch size was used during training"""

    dyn_input_shape = tf.shape(value)
    batch_size = dyn_input_shape[0]
    output_shape = tf.stack([batch_size, output_shape[1], output_shape[2]])

    return tf.contrib.nn.conv1d_transpose(
        value,
        filter,
        output_shape,
        stride,
        padding=padding,
        data_format=data_format,
        name=name
    )

def MakeBoundaryLoss(Geometry_tensor, boundary):
    """
    Make the boundary loss using boundary given
    :param Geometry_tensor: 8 element geometry h0 h1 h2 h3 r0 r1 r2 r3
    :param boundary: 4 element numpy array representing [h_low, h_high, r_low, r_high]
    return Boundary_loss: loss that depend on the boundary loss
    """
    tolerance = 1
    print("Geometry_tensor_shape",Geometry_tensor.shape)
    #Make constants
    print(boundary[0] * np.ones([1,4]))
    h_low  = tf.constant((boundary[0] - tolerance) * np.ones([1,4]), name= 'h_low',dtype=tf.float32)
    h_high = tf.constant((boundary[1] + tolerance) * np.ones([1,4]), name= 'h_high',dtype=tf.float32)
    r_low  = tf.constant((boundary[2] - tolerance) * np.ones([1,4]), name= 'r_low',dtype=tf.float32)
    r_high = tf.constant((boundary[3] + tolerance) * np.ones([1,4]), name= 'r_high',dtype=tf.float32)
    
    #Get the 2 separate parts
    h = Geometry_tensor[:,0:4]
    r = Geometry_tensor[:,4:]
    zero = tf.constant(0,dtype=tf.float32,name='zero')
    
    print("shape of h:",h.shape)
    print("shape of r:",r.shape)
    print("shape of h_low:",h_low.shape)
    Boundary_loss = tf.reduce_sum(tf.math.maximum(zero, tf.math.subtract(h, h_high)) + tf.math.maximum(zero, tf.math.subtract(h_low, h) ) +\
                                  tf.math.maximum(zero, tf.math.subtract(r, r_high)) + tf.math.maximum(zero, tf.math.subtract(r_low, r) ))
    return Boundary_loss
#The backward model part
def my_model_backward(labels,  fc_filters,  reg_scale, conv1d_filters, filter_channel_list ):
    """
    My customized model function
    :param labels: input spectrum
    :param output_size: dimension of output data
    :return:
    """
    
    ##Record the variables before Backwardmodel is created
    BeforeBackCollectionName = "BeforeBack_Collection"
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      tf.add_to_collection(BeforeBackCollectionName, var)
    print("Before Backward Model there is:",tf.get_collection(BeforeBackCollectionName))
    
    ##Building the model
    with tf.name_scope("BackwardModel"):
      print("Before convolution:", labels)
      preConv = tf.expand_dims(labels, axis=2)
      for cnt, (filters_length, filter_channels) in enumerate(zip(conv1d_filters, filter_channel_list)):
          convf = tf.Variable(tf.random_normal([filters_length,  preConv.shape().as_list()[-1], filter_channels]))
          preConv = tf.nn.conv1d(preConv, convf, stride = 1, padding='VALID')
          print("At prev_conV level{} the precoV shape is {}".format(cnt, preConv.shape()))
      #preConv = tf.expand_dims(labels, axis=2)
      #3conv = tf.keras.layers.Conv1D(1, 2, strides = 2,padding = 'same',
      #                              activation = None, name = 'Conv1d')(preConv)
      #backward_fc = tf.squeeze(conv, axis=2)
      backward_fc = preConv
      print("After convolution:",backward_fc)
      for cnt, filters in enumerate(fc_filters):
          backward_fc = tf.layers.dense(inputs=backward_fc, units=filters, activation=tf.nn.leaky_relu, 
                                name='backward_fc{}'.format(cnt),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))
      backward_out = backward_fc
      merged_summary_op = tf.summary.merge_all()
      
    ##Take record of the variables that created
    BackCollectionName = "Backward_Model_Collection"
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      tf.add_to_collection(BackCollectionName, var)
    print("Backward_out.shape", backward_out.shape)
    return backward_out, merged_summary_op, BackCollectionName, BeforeBackCollectionName

def my_model_fn_tens(backward_out, features, batch_size, clip,
                   fc_filters, tconv_fNums, tconv_dims, tconv_filters,
                   n_filter, n_branch, reg_scale, 
                     BackCollectionName, boundary):
    """
    My customized model function
    :param features: input features
    :param output_size: dimension of output data
    :return:
    """
    #Make a condition that if variable is True, train from feature
    print("backward_out.shape", backward_out.shape)
    print("features.shape",features.shape)
	
    train_Forward = tf.get_variable("train_forward",[],dtype = tf.bool,
                                       initializer = tf.zeros_initializer(),trainable =False)
    forward_in = tf.cond(train_Forward, true_fn= lambda: features, false_fn= lambda: backward_out);
    #Make the Boundary Loss
    Boundary_loss = MakeBoundaryLoss(forward_in, boundary)
    fc = forward_in
    for cnt, filters in enumerate(fc_filters):
        fc = tf.layers.dense(inputs=fc, units=filters, activation=tf.nn.leaky_relu, name='fc{}'.format(cnt),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))
    preTconv = fc
    tf.summary.histogram("preTconv", preTconv[0])  # select 0th element or else histogram reduces the batch
    up = tf.expand_dims(preTconv, axis=2)
    feature_dim = fc_filters[-1]
    
    last_filter = 1
    for cnt, (up_fNum, up_size, up_filter) in enumerate(zip(tconv_fNums, tconv_dims, tconv_filters)):
        assert up_size % feature_dim == 0, "up_size={} while feature_dim={} (cnt={})! " \
                                        "Thus mod is {}".format(up_size, feature_dim, cnt, up_size%feature_dim)
        stride = up_size // feature_dim
        feature_dim = up_size
        f = tf.Variable(tf.random_normal([up_fNum, up_filter, last_filter]))
        up = conv1d_transpose_wrap(up, f, [batch_size, up_size, up_filter], stride, name='up{}'.format(cnt))
        last_filter = up_filter

    preconv = up
    up = tf.layers.conv1d(preconv, 1, 1, activation=None, name='conv_final')
    up = up[:, clip:-clip]
    up = tf.squeeze(up, axis=2)
    # up = tf.layers.dense(inputs=up, units=tconv_dims[-1], activation=tf.nn.leaky_relu, name='fc_final',
    #                           kernel_initializer=tf.random_normal_initializer(stddev=0.02))
    merged_summary_op = tf.summary.merge_all()
    
    ##Get a collection of variables that created in forward model
    ForwardCollectionName = "Forward_Model_Collection"
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      if var not in tf.get_collection(BackCollectionName):
        tf.add_to_collection(ForwardCollectionName, var)
    return forward_in, up, preconv, preTconv, merged_summary_op, ForwardCollectionName, train_Forward, Boundary_loss
  
def tandem_model(features,labels, backward_fc,   batch_size, clip,
                 fc_filters, tconv_fNums, tconv_dims, tconv_filters,
                 n_filter, n_branch, reg_scale, boundary, conv1d_filters, filter_channel_list):
    """
    Customized tandem model which combines 2 model
    """
    backward_out, summary_out,BackCollectionName, BeforeBackCollectionName =\
                          my_model_backward(labels, backward_fc, reg_scale, conv1d_filters,filter_channel_list)
    forward_in, up, preconv, preTconv,merged_summary_op, ForwardCollectionName, train_Forward, Boundary_loss = \
                          my_model_fn_tens(backward_out,features,batch_size, clip,
                                            fc_filters, tconv_fNums, tconv_dims, tconv_filters,
                                            n_filter, n_branch, reg_scale, BackCollectionName, boundary)
    return forward_in, up, merged_summary_op, ForwardCollectionName,\
            BackCollectionName, backward_out, train_Forward, Boundary_loss

