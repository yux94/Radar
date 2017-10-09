# -*- coding: utf-8 -*-
import tensorflow as tf

class Decoupling_Net():
    
    def __init__(self):
        return  
    
    def _variable_decay(self,var,wd):
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=var.op.name+'_loss')
        tf.add_to_collection('loss', weight_decay)
        return
        
    def _get_fc_weight(self,name,shape,initializer):
        weight = tf.get_variable(name=name+'_weights',shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=tf.float32),
                            dtype=tf.float32)
        return weight                    
    
    
    def _get_conv_filter(self,name,shape,initializer):  
        
        weight = tf.get_variable(name=name+'_weight',shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=tf.float32),
                            dtype=tf.float32)
            
        return weight
    
    def _get_bias(self,name,shape,initializer):
        bias = tf.get_variable(name=name+'_biases',shape=shape,
                        initializer=tf.constant_initializer(value=initializer, dtype=tf.float32),
                            dtype=tf.float32)
        return bias
    
    def _conv_layer(self,Bottom,ks,num_output,initializer,stride,pad,name):
        #shape=(filiter_dim,1,in_channels,out_channels),Bottom=[batch, in_width , 1 , in_channels]
        in_channels = Bottom.get_shape().as_list()[3]
        shape=(ks,1,in_channels,num_output)
        weight = self._get_conv_filter(name,shape,initializer[0])
        conv_biases = self._get_bias(name,shape[-1],initializer[1])    
        
        conv = tf.nn.conv2d(Bottom, weight, stride, padding=pad)
        bias = tf.nn.bias_add(conv, conv_biases)
        
        if self.Phase == 'Train':
            self._variable_decay(weight,0.00005)
            self._variable_decay(conv_biases,0.0)        
        
        return bias
    
    
    def _ReLU(self,Bottom):
        relu = tf.nn.relu(Bottom)
        return relu
    
    def _BatchNormalization(self,Bottom,name):
        
        axis = list(range(len(Bottom.get_shape())-1))
        mean,variance = tf.nn.moments(Bottom,axis)
    
        Beta = tf.get_variable(name=name+'_offset',
                        initializer=tf.zeros_initializer(shape=mean.get_shape().as_list(),dtype=tf.float32),
                            dtype=tf.float32)
        Gamma =  tf.get_variable(name=name+'_scale',
                        initializer=tf.ones_initializer(shape=mean.get_shape().as_list(), dtype=tf.float32),
                            dtype=tf.float32)                   
        moving_mean =  tf.get_variable(name=name+'_moving_mean',
                        initializer=tf.zeros_initializer(shape=mean.get_shape().as_list(),dtype=tf.float32),
                            dtype=tf.float32,trainable=False)    
        moving_variance = tf.get_variable(name=name+'_moving_variance',
                        initializer=tf.ones_initializer(shape=mean.get_shape().as_list(), dtype=tf.float32),
                            dtype=tf.float32,trainable=False)   
                            
        if self.Phase == 'Train':
             moving_mean -= (1 - 0.9) * (moving_mean - mean)  
             moving_variance -= (1 - 0.9) * (moving_variance - variance)
             bn = tf.nn.batch_normalization(Bottom,moving_mean,moving_variance,Beta,Gamma,1e-5)
        else:
             bn = tf.nn.batch_normalization(Bottom,moving_mean,moving_variance,Beta,Gamma,1e-5)
             
        return bn

    def _Memory_Module(self,Bottom,label,Memory_size,feature_size,k,name):
        
        Memory_Space = tf.get_variable(name='Memory_Space',shape=[Memory_size,feature_size],
                        initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32),
                            dtype=tf.float32)
       
        Memory_Value = tf.get_variable(name='Memory_Value',
                        initializer=tf.zeros_initializer(shape=Memory_size,dtype=tf.float32),
                            dtype=tf.float32)        
        
        Memory_Age = tf.get_variable(name='Memory_Age',
                        initializer=tf.zeros_initializer(shape=Memory_size,dtype=tf.float32),
                            dtype=tf.float32)
                            
        Memory_Space = tf.nn.l2_normalize(Memory_Space,1)            
        Bottom_Normalize = tf.nn.l2_normalize(Bottom,1)
        
        
        return

    
    def Build_CNN(self,Bottom,label,Phase) :
        self.Phase = Phase
        
        batchsize = Bottom.get_shape().as_list()[0]
        dim = Bottom.get_shape().as_list()[1] 
        Bottom = tf.reshape(Bottom,[batchsize,dim,1,1])        

        '''begin convolutional layer'''
        conv1 = self._conv_layer(Bottom,32,128,(0.01,0.0),(1,2,1,1),'SAME','conv1')
        relu1  = self._ReLU(conv1)
        
        max_pool1 = tf.nn.max_pool(relu1,(1,4,1,1),(1,2,1,1),'VALID')#256
        
        ''' convolutional layer2 '''
        conv2 = self._conv_layer(max_pool1,8,256,(0.01,0.0),(1,1,1,1),'SAME','conv2')
        relu2  = self._ReLU(conv2)
        max_pool2 = tf.nn.max_pool(relu2,(1,4,1,1),(1,2,1,1),'VALID')#128

        ''' convolutional layer3 '''
        conv3 = self._conv_layer(max_pool2,8,256,(0.01,0.0),(1,1,1,1),'SAME','conv3')
        relu3  = self._ReLU(conv3)
        
        max_pool3 = tf.nn.max_pool(relu3,(1,4,1,1),(1,2,1,1),'VALID')#64       

        ''' convolutional layer4 '''
        conv4 = self._conv_layer(max_pool3,4,512,(0.01,0.0),(1,1,1,1),'SAME','conv4')
        relu4  = self._ReLU(conv4)
        max_pool4 = tf.nn.max_pool(relu4,(1,4,1,1),(1,2,1,1),'VALID')#32  

        ''' convolutional layer5 '''
        conv5 = self._conv_layer(max_pool4,4,1024,(0.01,0.0),(1,1,1,1),'SAME','conv5')
        relu5  = self._ReLU(conv5)
        
        
        max_pool5 = tf.nn.max_pool(relu5,(1,4,1,1),(1,2,1,1),'VALID')#16  
        
        resize = self._reshape_conv_to_fc(max_pool5)
        
        fc1 = self._fully_connect_layer(resize,1024,(0.005,1.0),'ReLU','fc1')
        fc2 = self._fully_connect_layer(fc1,1024,(0.005,1.0),'ReLU','fc2')
        
        

        
        return 
