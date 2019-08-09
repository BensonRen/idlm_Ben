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
  
#Define the back-prop mmoel
def back_prop_model(features, batch_size, clip,
                   fc_filters, tconv_fNums, tconv_dims, tconv_filters,
                   n_filter, n_branch, reg_scale, boundary):
    """
    Customized model for using back-propagation
    Use a extra variable for the place of 
    """
    #Make the variable geometry
    geometry_variable = tf.get_variable("Geometry_var", shape= features.shape, dtype = tf.float32, 
                                        initializer = tf.zeros_initializer(), trainable = True)
    
    #Make a condition that if variable is True, train from feature
    train_Forward = tf.get_variable("train_forward",[],dtype = tf.bool,
                                       initializer = tf.zeros_initializer(),trainable =False)
    
    forward_in = tf.cond(train_Forward, true_fn= lambda: features, false_fn= lambda: geometry_variable)
    
    #Make the Boundary Loss
    Boundary_loss = MakeBoundaryLoss(forward_in, boundary)
    
    fc = forward_in
    
    #print("Backward_Out:", backward_out)
    #print("features:", features)
    #print("FC layer:",fc)
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
    return forward_in, up, merged_summary_op, geometry_variable, train_Forward, Boundary_loss
  

def initializeInBoundary(shape, boundary):
    """
    Initialize a np array within the boundary
    """
    RN = np.random.random(size = shape)
    RN[:,0:4] = RN[:,0:4] * (boundary[1] - boundary[0]) +boundary[0]
    RN[:,4:] = RN[:,4:] * (boundary[3] - boundary[2]) +boundary[2]
    
    #print(RN)
    return RN
