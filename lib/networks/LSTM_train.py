import tensorflow as tf
from .network import Network
from ..lstm.config import cfg


class LSTM_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        '''
        self.data = tf.placeholder(tf.float32, shape=[64, 400, cfg.NUM_FEATURES],
                                   name='data')  # N*t_s*features*channels
        self.labels = tf.placeholder(tf.int32, [972], name='labels')
        self.time_step_len = tf.placeholder(tf.int32, [64], name='time_step_len')
        self.labels_len = tf.placeholder(tf.int32, [64], name='labels_len')
        '''

        self.data = tf.placeholder(tf.float32, shape=[None, None, cfg.NUM_FEATURES ], name='data') #N*t_s*features*channels
        self.labels = tf.placeholder(tf.int32,[None],name='labels')
        self.time_step_len = tf.placeholder(tf.int32,[None], name='time_step_len')
        self.labels_len = tf.placeholder(tf.int32,[None],name='labels_len')

        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data,'labels':self.labels,
                            'time_step_len':self.time_step_len,
                            'labels_len':self.labels_len})
        self.trainable = trainable
        self.setup()

    def setup(self):
        self.feed('data')
        self.conv_single(3, 3, 64, 1, 1, name='conv1', c_i=cfg.NCHANNELS)
        self.max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
        self.conv_single(3, 3, 128, 1, 1, name='conv2')
        self.max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
        self.conv_single(3, 3, 256, 1, 1, name='conv3_1')
        self.conv_single(3, 3, 256, 1, 1, name='conv3_2')
        self.max_pool(1, 2, 1, 2, padding='VALID', name='pool2')
        self.conv_single(3, 3, 512, 1, 1, name='conv4_1', bn=True)
        self.conv_single(3, 3, 512, 1, 1, name='conv4_2', bn=True)
        self.max_pool(1, 2, 1, 2, padding='VALID', name='pool3')
        self.conv_single(2, 2, 512, 1, 1, padding='VALID', name='conv5', relu=False)
        # self.dropout(keep_prob = self.keep_prob, name = 'dropout_layer')
        self.reshape_squeeze_layer(d=512, name='reshaped_layer')
        self.feed('reshaped_layer', 'time_step_len')
        self.bi_lstm(cfg.TRAIN.NUM_HID, cfg.TRAIN.NUM_LAYERS, name='logits')
            # .lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))
            # .bi_lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))


'''
    def setup(self):
        (self.feed('data')
         .conv_single(3, 3, 64 ,1, 1, name='conv1',c_i=cfg.NCHANNELS)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv_single(3, 3, 128 ,1, 1, name='conv2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv_single(3, 3, 256 ,1, 1, name='conv3_1')
         .conv_single(3, 3, 256 ,1, 1, name='conv3_2')
         .max_pool(1, 2, 1, 2, padding='VALID', name='pool2')
         .conv_single(3, 3, 512 ,1, 1, name='conv4_1', bn=True)
         .conv_single(3, 3, 512 ,1, 1, name='conv4_2', bn=True)
         .max_pool(1, 2, 1, 2, padding='VALID', name='pool3')
         .conv_single(2, 2, 512 ,1, 1, padding = 'VALID', name='conv5', relu=False)
         #.dropout(keep_prob = self.keep_prob, name = 'dropout_layer')
         .reshape_squeeze_layer(d = 512 , name='reshaped_layer'))
        (self.feed('reshaped_layer','time_step_len')
         .bi_lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits'))
         # .lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))
         #.bi_lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))
'''