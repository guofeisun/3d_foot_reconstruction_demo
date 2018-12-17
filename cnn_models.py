import tensorflow as tf
from ops import Ops
from tensorflow.contrib import rnn
        
def get_rnn_cell(unit_size=2000):
    return rnn.BasicLSTMCell(unit_size)
def get_rnn3d_cell():
    return rnn.ConvLSTMCell(3, [4,4,4,512], 512, [1,1,1])
def encoder(X, volume_shape, flag='normal'):
    X = tf.reshape(X,[-1, volume_shape[0], volume_shape[1], volume_shape[2], 1])
    if flag == 'small':
        c_e = [1,32,64,128,256]
    else:
        c_e = [1,64,128,256,512]
    s_e = [0, 1, 1, 1, 1]
    layers_e = []
    layers_e.append(X)
    for i in range(1,5,1):
        layer = Ops.conv3d(layers_e[-1],k=4,out_c=c_e[i],str=s_e[i],name='e'+str(i))
        layer = Ops.maxpool3d(Ops.xxlu(layer, label='lrelu'), k=2,s=2,pad='SAME')
        layers_e.append(layer)

    ### fc
    [_, d1, d2, d3, cc] = layers_e[-1].get_shape()
    d1=int(d1); d2=int(d2); d3=int(d3); cc=int(cc)
    lfc = tf.reshape(layers_e[-1],[-1, int(d1)*int(d2)*int(d3)*int(cc)])
    if flag == 'small':
        lfc = Ops.xxlu(Ops.fc(lfc, out_d=1000,name='fc1'), label='relu')    
    else:
        lfc = Ops.xxlu(Ops.fc(lfc, out_d=2000,name='fc1'), label='relu')

    return lfc, int(d1), int(d2), int(d3), int(cc)
def encoder_no_fc(X, volume_shape, flag='normal'):
    X = tf.reshape(X,[-1, volume_shape[0], volume_shape[1], volume_shape[2], 1])
    if flag == 'small':
        c_e = [1,32,64,128,256]
    else:
        c_e = [1,64,128,256,512]
    s_e = [0, 1, 1, 1, 1]
    layers_e = []
    layers_e.append(X)
    for i in range(1,5,1):
        layer = Ops.conv3d(layers_e[-1],k=4,out_c=c_e[i],str=s_e[i],name='e'+str(i))
        layer = Ops.maxpool3d(Ops.xxlu(layer, label='lrelu'), k=2,s=2,pad='SAME')
        layers_e.append(layer)

    return layers_e[-1]
def decoder(lfc, d1, d2, d3, cc, flag='normal'):
    lfc = Ops.xxlu(Ops.fc(lfc,out_d=d1*d2*d3*cc, name='fc2'), label='relu')
    lfc = tf.reshape(lfc, [-1, d1,d2,d3,cc])

    if flag == 'small':
        c_d = [0,128,64,32,16,8]
    else:
        c_d = [0,256,128,64,16,8]
    s_d = [0,2,2,2,2,2]
    layers_d = []
    layers_d.append(lfc)
    for j in range(1,6,1):
        layer = Ops.deconv3d(layers_d[-1],k=4,out_c=c_d[j],str=s_d[j],name='d'+str(len(layers_d)))

        layer = Ops.xxlu(layer, label='relu')
        layers_d.append(layer)
    ###
    layer = Ops.deconv3d(layers_d[-1],k=4,out_c=1,str=2,name='dlast')
    ###
    Y_sig = tf.nn.sigmoid(layer)
    Y_sig_modi = tf.maximum(Y_sig,0.01)

    return Y_sig, Y_sig_modi
def rnn_fusion(v, initial_state, unit_size=2000):
    lstm_cell = get_rnn_cell(unit_size)
    outputs = []
    states = []
    states.append(initial_state)
    rnn_step = len(v)
    for i in range(rnn_step):
        tf.split
        output, state = lstm_cell(v[i], states[-1])
        outputs.append(output)
        states.append(state)
    return outputs[-1], state
def rnn3d_fusion(v, initial_state):
    lstm_cell = get_rnn3d_cell()
    outputs = []
    states = []
    states.append(initial_state)
    rnn_step = len(v)
    for i in range(rnn_step):
        tf.split
        output, state = lstm_cell(v[i], states[-1])
        outputs.append(output)
        states.append(state)
    return outputs[-1], state
def aeu_multiview(X_f, X_r, X_b, X_l, volume_shape, flag='normal'):
    with tf.variable_scope('encoder'):
        v_f, d1, d2, d3, cc = encoder(X_f, volume_shape, flag)
    with tf.variable_scope('encoder', reuse=True):
        v_r, _, _, _, _ = encoder(X_r, volume_shape, flag)
    with tf.variable_scope('encoder', reuse=True):
        v_b, _, _, _, _ = encoder(X_b, volume_shape, flag)
    with tf.variable_scope('encoder', reuse=True):
        v_l, _, _, _, _ = encoder(X_l, volume_shape, flag)
    lfc = tf.reduce_max([v_f, v_r, v_b, v_l], axis=0)
    with tf.variable_scope('decoder'):
        Y_sig, Y_sig_modi = decoder(lfc, d1, d2, d3, cc, flag)
    
    return Y_sig, Y_sig_modi
def aeu_orientation_pooling(X_f, X_r, X_b, X_l, volume_shape, flag='normal'):
    with tf.variable_scope('encoder_no_fc'):
        v_f = encoder_no_fc(X_f, volume_shape, flag)
    with tf.variable_scope('encoder_no_fc', reuse=True):
        v_r = encoder_no_fc(X_r, volume_shape, flag)
    with tf.variable_scope('encoder_no_fc', reuse=True):
        v_b = encoder_no_fc(X_b, volume_shape, flag)
    with tf.variable_scope('encoder_no_fc', reuse=True):
        v_l = encoder_no_fc(X_l, volume_shape, flag)
    v = tf.reduce_max([v_f, v_r, v_b, v_l], axis=0)

    ### fc
    [_, d1, d2, d3, cc] = v.get_shape()
    d1=int(d1); d2=int(d2); d3=int(d3); cc=int(cc)
    lfc = tf.reshape(v, [-1, int(d1)*int(d2)*int(d3)*int(cc)])
    if flag == 'small':
        lfc = Ops.xxlu(Ops.fc(lfc, out_d=1000,name='fc1'), label='relu')    
    else:
        lfc = Ops.xxlu(Ops.fc(lfc, out_d=2000,name='fc1'), label='relu')
    
    with tf.variable_scope('decoder'):
        Y_sig, Y_sig_modi = decoder(lfc, d1, d2, d3, cc, flag)
    
    return Y_sig, Y_sig_modi
def aeu_rnn_fusion(X_f, X_r, X_b, X_l, volume_shape, initial_state, flag='normal'):
    with tf.variable_scope('encoder'):
        v_f, d1, d2, d3, cc = encoder(X_f, volume_shape, flag)
    with tf.variable_scope('encoder', reuse=True):
        v_r, _, _, _, _ = encoder(X_r, volume_shape, flag)
    with tf.variable_scope('encoder', reuse=True):
        v_b, _, _, _, _ = encoder(X_b, volume_shape, flag)
    with tf.variable_scope('encoder', reuse=True):
        v_l, _, _, _, _ = encoder(X_l, volume_shape, flag)
    # rnn fusion
    with tf.variable_scope('rnn_fusion'):
        v_fused, _= rnn_fusion([v_f, v_r, v_b, v_l], initial_state, 2000)
        v_fused = Ops.xxlu(v_fused, label='relu')
    with tf.variable_scope('decoder'):
        Y_sig, Y_sig_modi = decoder(v_fused, d1, d2, d3, cc, flag)
    
    return Y_sig, Y_sig_modi
def aeu_rnn3d_fusion(X_f, X_r, X_b, X_l, volume_shape, initial_state, flag='normal'):
    with tf.variable_scope('encoder_no_fc'):
        v_f = encoder_no_fc(X_f, volume_shape, flag)
    with tf.variable_scope('encoder_no_fc', reuse=True):
        v_r = encoder_no_fc(X_r, volume_shape, flag)
    with tf.variable_scope('encoder_no_fc', reuse=True):
        v_b = encoder_no_fc(X_b, volume_shape, flag)
    with tf.variable_scope('encoder_no_fc', reuse=True):
        v_l = encoder_no_fc(X_l, volume_shape, flag)
    # 3drnn fusion
    with tf.variable_scope('rnn3d_fusion'):
        v_fused, _= rnn3d_fusion([v_f, v_r, v_b, v_l], initial_state)
        v_fused = Ops.xxlu(v_fused, label='lrelu')
    ### fc
    [_, d1, d2, d3, cc] = v_fused.get_shape()
    d1=int(d1); d2=int(d2); d3=int(d3); cc=int(cc)
    lfc = tf.reshape(v_fused, [-1, int(d1)*int(d2)*int(d3)*int(cc)])
    lfc = Ops.xxlu(Ops.fc(lfc, out_d=2000,name='fc1'), label='relu')
    with tf.variable_scope('decoder'):
        Y_sig, Y_sig_modi = decoder(lfc, d1, d2, d3, cc, flag)
    
    return Y_sig, Y_sig_modi
def aeu(X, volume_shape):
    X = tf.reshape(X,[-1, volume_shape[0], volume_shape[1], volume_shape[2], 1])
    c_e = [1,64,128,256,512]
    s_e = [0, 1, 1, 1, 1]
    layers_e = []
    layers_e.append(X)
    for i in range(1,5,1):
        layer = Ops.conv3d(layers_e[-1],k=4,out_c=c_e[i],str=s_e[i],name='e'+str(i))
        layer = Ops.maxpool3d(Ops.xxlu(layer, label='lrelu'), k=2,s=2,pad='SAME')
        layers_e.append(layer)

    ### fc
    [_, d1, d2, d3, cc] = layers_e[-1].get_shape()
    d1=int(d1); d2=int(d2); d3=int(d3); cc=int(cc)
    lfc = tf.reshape(layers_e[-1],[-1, int(d1)*int(d2)*int(d3)*int(cc)])
    lfc = Ops.xxlu(Ops.fc(lfc, out_d=2000,name='fc1'), label='relu')

    lfc = Ops.xxlu(Ops.fc(lfc,out_d=d1*d2*d3*cc, name='fc2'), label='relu')
    lfc = tf.reshape(lfc, [-1, d1,d2,d3,cc])

    c_d = [0,256,128,64,16,8]
    s_d = [0,2,2,2,2,2]
    layers_d = []
    layers_d.append(lfc)
    for j in range(1,6,1):
        u_net = True
        if u_net:
            layer = tf.concat([layers_d[-1], layers_e[-j]],axis=4)
            layer = Ops.deconv3d(layer, k=4,out_c=c_d[j], str=s_d[j],name='d'+str(len(layers_d)))
        else:
            layer = Ops.deconv3d(layers_d[-1],k=4,out_c=c_d[j],str=s_d[j],name='d'+str(len(layers_d)))

        layer = Ops.xxlu(layer, label='relu')
        layers_d.append(layer)
    ###
    layer = Ops.deconv3d(layers_d[-1],k=4,out_c=1,str=2,name='dlast')
    ###
    Y_sig = tf.nn.sigmoid(layer)
    Y_sig_modi = tf.maximum(Y_sig,0.01)

    return Y_sig, Y_sig_modi

def dis(X, Y, X_shape, Y_shape, is_conditional=True):
    X = tf.reshape(X,[-1, X_shape[0], X_shape[1], X_shape[2], 1])
    X = tf.reshape(X, [-1, Y_shape[0], Y_shape[1], 4, 1])
    Y = tf.reshape(Y,[-1, Y_shape[0], Y_shape[1], Y_shape[2], 1])
    if is_conditional:
        Y = tf.concat([X, Y], axis=3)

    c_d = [1,8,16,32,64,128,256]
    s_d = [0,2,2,2,2,2,2]
    layers_d =[]
    layers_d.append(Y)
    for i in range(1,7,1):
        layer = Ops.conv3d(layers_d[-1],k=4,out_c=c_d[i],str=s_d[i],name='d'+str(i))
        if i!=6:
            layer = Ops.xxlu(layer, label='lrelu')
        layers_d.append(layer)
    [_, d1, d2, d3, cc] = layers_d[-1].get_shape()
    d1 = int(d1); d2 = int(d2); d3 = int(d3); cc = int(cc)
    y = tf.reshape(layers_d[-1],[-1,d1*d2*d3*cc])
    return tf.nn.sigmoid(y)