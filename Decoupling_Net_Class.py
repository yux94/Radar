# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class Decoupling_Net():
    
    def __init__(self):
        return  
    
    def _variable_decay(self,var,wd):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=var.op.name+'_loss')
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
    
    def _Dropout(self,bottom):
        if self.Phase=='Train':
            bottom = tf.nn.dropout(bottom,keep_prob=0.5)
        else:
            bottom = tf.nn.dropout(bottom,keep_prob=1.0)
        return bottom
    
    def layer_norm(self,Bottom,eps,name):
        
        axis = list(range(len(Bottom.get_shape())-1))
        mean,var = tf.nn.moments(Bottom,axis)
        
        gain = tf.get_variable(name=name+'_gain',
                         shape=mean.get_shape().as_list(), initializer=tf.ones_initializer(),dtype=tf.float32) 
        #        initializer=tf.zeros_initializer(shape=mean.get_shape().as_list(),dtype=tf.float32),
#                            dtype=tf.float32)
        bias =  tf.get_variable(name=name+'_scale',
                        shape=mean.get_shape().as_list(), initializer=tf.zeros_initializer(),dtype=tf.float32) 
        
        ln = (Bottom - mean)/tf.sqrt(var + eps)
        
        return ln * gain + bias
        
    def _BatchNormalization(self,Bottom,name):
        
        axis = list(range(len(Bottom.get_shape())-1))
        mean,variance = tf.nn.moments(Bottom,axis)
    
        Beta = tf.get_variable(name=name+'_offset',
                         shape=mean.get_shape().as_list(), initializer=tf.zeros_initializer(),dtype=tf.float32) 
        #        initializer=tf.zeros_initializer(shape=mean.get_shape().as_list(),dtype=tf.float32),
#                            dtype=tf.float32)
        Gamma =  tf.get_variable(name=name+'_scale',
                        shape=mean.get_shape().as_list(), initializer=tf.ones_initializer(),dtype=tf.float32) 
#                        initializer=tf.ones_initializer(shape=mean.get_shape().as_list(), dtype=tf.float32),
#                            dtype=tf.float32)                   
        moving_mean =  tf.get_variable(name=name+'_moving_mean',                        
                        shape=mean.get_shape().as_list(), initializer=tf.zeros_initializer(),dtype=tf.float32,trainable=False)
#                        initializer=tf.zeros_initializer(shape=mean.get_shape().as_list(),dtype=tf.float32),
#                            dtype=tf.float32,trainable=False)    
        moving_variance = tf.get_variable(name=name+'_moving_variance',
                        shape=mean.get_shape().as_list(), initializer=tf.ones_initializer(),dtype=tf.float32,trainable=False)
#                        initializer=tf.ones_initializer(shape=mean.get_shape().as_list(), dtype=tf.float32),
#                            dtype=tf.float32,trainable=False)   
                            
        if self.Phase == 'Train':
             moving_mean -= (1 - 0.9) * (moving_mean - mean)  
             moving_variance -= (1 - 0.9) * (moving_variance - variance)
             bn = tf.nn.batch_normalization(Bottom,moving_mean,moving_variance,Beta,Gamma,1e-5)
#             bn = tf.nn.batch_normalization(Bottom,mean,variance,Beta,Gamma,1e-5)

        else:
             bn = tf.nn.batch_normalization(Bottom,moving_mean,moving_variance,Beta,Gamma,1e-5)
             
        return bn

    def _fully_connect_layer(self,Bottom,output_num,initializer,active,name):
        
        dim = Bottom.get_shape().as_list()[1]
        shape = (dim,output_num)
        weight = self._get_fc_weight(name,shape,initializer[0])
        biases = self._get_bias(name,output_num,initializer[1])
        fc = tf.nn.bias_add(tf.matmul(Bottom,weight),biases)
        
        if active == 'ReLU':
            fc = tf.nn.relu(fc)
        
        if self.Phase == 'Train':
            self._variable_decay(weight,0.00005)
            self._variable_decay(biases,0.0)        
        
        return fc
        
    def _reshape_conv_to_fc(self,bottom):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
                 dim *= d
        x = tf.reshape(bottom, [-1, dim])
        return x    
        
    
    def Build_CNN(self,Bottom,Phase) :
        self.Phase = Phase
        
        batchsize = Bottom.get_shape().as_list()[0]
        dim = Bottom.get_shape().as_list()[1] 
        Bottom = tf.reshape(Bottom,[batchsize,dim,1,1])        

        '''begin convolutional layer'''
        conv1 = self._conv_layer(Bottom,32,128,(0.01,0.0),(1,2,1,1),'SAME','conv1')
        bn1 = self._BatchNormalization(conv1,'bn1_1')
#        bn1 = self.layer_norm(conv1,1e-5,'ln1_1')
        relu1  = self._ReLU(bn1)
#        relu1  = self._ReLU(conv1)
        max_pool1 = tf.nn.max_pool(relu1,(1,4,1,1),(1,2,1,1),'VALID')#256
        
        ''' convolutional layer2 '''
        conv2 = self._conv_layer(max_pool1,8,256,(0.01,0.0),(1,1,1,1),'SAME','conv2')
        bn2 = self._BatchNormalization(conv2,'bn2_1')
#        bn2 = self.layer_norm(conv2,1e-5,'ln2_1')
        relu2  = self._ReLU(bn2)
#        relu2  = self._ReLU(conv2)

        max_pool2 = tf.nn.max_pool(relu2,(1,4,1,1),(1,2,1,1),'VALID')#128

        ''' convolutional layer3 '''
        conv3 = self._conv_layer(max_pool2,8,256,(0.01,0.0),(1,1,1,1),'SAME','conv3')
        bn3 = self._BatchNormalization(conv3,'bn3_1')
#        bn3 = self.layer_norm(conv3,1e-5,'ln3_1')
        relu3  = self._ReLU(bn3)
#        relu3  = self._ReLU(conv3)
        
        max_pool3 = tf.nn.max_pool(relu3,(1,4,1,1),(1,2,1,1),'VALID')#64       

        ''' convolutional layer4 '''
        conv4 = self._conv_layer(max_pool3,4,512,(0.01,0.0),(1,1,1,1),'SAME','conv4')
        bn4 = self._BatchNormalization(conv4,'bn4_1')
#        bn4 = self.layer_norm(conv4,1e-5,'ln4_1')
        relu4  = self._ReLU(bn4)
#        relu4  = self._ReLU(conv4)
#        
        max_pool4 = tf.nn.max_pool(relu4,(1,4,1,1),(1,2,1,1),'VALID')#32  

        ''' convolutional layer5 '''
        conv5 = self._conv_layer(max_pool4,4,1024,(0.01,0.0),(1,1,1,1),'SAME','conv5')
        bn5 = self._BatchNormalization(conv5,'bn5_1')
#        bn5 = self.layer_norm(conv5,1e-5,'ln5_1')
        relu5  = self._ReLU(bn5)
#        relu5  = self._ReLU(conv5)
        
        
        max_pool5 = tf.nn.max_pool(relu5,(1,4,1,1),(1,2,1,1),'VALID')#16  
        
        resize = self._reshape_conv_to_fc(max_pool5)
        
        fc1 = self._fully_connect_layer(resize,4096,(0.005,1.0),'ReLU','fc1')
        fc2 = self._fully_connect_layer(fc1,4096,(0.005,1.0),'ReLU','fc2')
#        fc2 = self._Dropout(fc2)
# 
        identity_feature = self._fully_connect_layer(fc2,1024,(0.005,1.0),'ReLU','identity_feature')##########
        label_output = self._fully_connect_layer(identity_feature,3,(0.005,0.0),'None','label_output')
        
        theta_feature = self._fully_connect_layer(fc2,1024,(0.005,1.0),'ReLU','theata_feature')############
        theta_output = self._fully_connect_layer(theta_feature,2,(0.005,0.0),'None','theata_output')        
#        
#        
        output_list = [label_output,theta_output]
        feature_list = [identity_feature,theta_feature]
        
        return output_list,feature_list
        
    def Mix_Feature_Angle(self,Bottom,seed=0,num_planes=3,Phase='Train'):
        
        if Phase == 'Train':
            all_output = []
            all_feature = []        
            with tf.variable_scope('pub') as scope:
                first_output,first_feature = self.Build_CNN(Bottom[0],Phase)
                all_output.append(first_output)
                all_feature.append(first_feature)
                
                scope.reuse_variables() 
                for i in range(1,num_planes):
                    output,feature = self.Build_CNN(Bottom[i],Phase)
                    all_output.append(output)
                    all_feature.append(feature)
            
            all_sample_identity_feature = all_feature[0][0]
            all_sample_theta_feature = all_feature[0][1]
            for i in range(1,len(all_feature)):
                all_sample_identity_feature = tf.concat([all_sample_identity_feature,all_feature[i][0]],0)
                all_sample_theta_feature = tf.concat([all_sample_theta_feature,all_feature[i][1]],0)
            
            Angle_perturbation = tf.stop_gradient(tf.random_shuffle(all_sample_theta_feature,seed))
            Mix_Feature = tf.concat([all_sample_identity_feature,Angle_perturbation],1)
            
            Mix_fc1 = self._fully_connect_layer(Mix_Feature,1024,(0.005,0.0),'ReLU','Mix_fc1')
            Mix_fc2 = self._fully_connect_layer(Mix_fc1,1024,(0.005,0.0),'ReLU','Mix_fc2')
            label_output = self._fully_connect_layer(Mix_fc2,3,(0.005,0.0),'None','mix_label_output')
            theta_output = self._fully_connect_layer(Mix_fc2,2,(0.005,0.0),'None','mix_theata_output')
            Mix_output = [label_output,theta_output]
            return all_feature,all_output,Mix_output
        
        else:
            with tf.variable_scope('pub') as scope:
                output_list,feature_list = self.Build_CNN(Bottom,Phase)
            Mix_Feature = tf.concat([feature_list[0],feature_list[1]],1)
            Mix_fc1 = self._fully_connect_layer(Mix_Feature,1024,(0.005,0.0),'ReLU','Mix_fc1')
            Mix_fc2 = self._fully_connect_layer(Mix_fc1,1024,(0.005,0.0),'ReLU','Mix_fc2')
            label_output = self._fully_connect_layer(Mix_fc2,3,(0.01,0.0),'None','mix_label_output')
            theta_output = self._fully_connect_layer(Mix_fc2,2,(0.005,1.0),'None','mix_theata_output')
            Mix_output = [label_output,theta_output]
            return feature_list,output_list,Mix_output
        
    def Detail_P6_ACC(self,Bottom,Labels,Labels_Angle):
        
#        feature,output,mix_output = self.Mix_Feature_Angle(Bottom,0,3,Phase='Test')             
        output,feature = self.Build_CNN(Bottom,Phase='Test')

#        y_pred = np.argmax(output[0], axis=-1)
#        y_pred_angle = np.argmax(output[1], axis=-1)  
#        y_pred = tf.cast(y_pred,dtype=tf.int64)
#        y_pred_angle = tf.cast(y_pred_angle,dtype=tf.int64)

        y_pred = output[0]
        y_pred_angle = output[1] 
        
#        y_pred = output[0]
#        y_pred_angle = output[1] 
        
        ACC_Right = tf.equal(tf.cast(tf.argmax(y_pred,1),dtype=tf.uint8), 
                                      tf.cast(Labels,dtype=tf.uint8))
        ACC_Right = tf.reduce_mean(tf.cast(ACC_Right, "float")) 
        
        ACC_Wrong_to_p2 = tf.equal(tf.cast(tf.argmax(y_pred,1),dtype=tf.uint8), 
                                      tf.cast(tf.zeros_like(Labels),dtype=tf.uint8))
        ACC_Wrong_to_p2 = tf.reduce_mean(tf.cast(ACC_Wrong_to_p2, "float"))        

        ACC_Wrong_to_p3 = tf.equal(tf.cast(tf.argmax(y_pred,1),dtype=tf.uint8), 
                                      tf.cast(tf.ones_like(Labels),dtype=tf.uint8))
        ACC_Wrong_to_p3 = tf.reduce_mean(tf.cast(ACC_Wrong_to_p3, "float"))   
        
                
        ACC_Right_theta = tf.equal(tf.cast(tf.argmax(y_pred_angle,1),dtype=tf.uint8), 
                                      tf.cast(Labels_Angle,dtype=tf.uint8))
        ACC_Right_theta = tf.reduce_mean(tf.cast(ACC_Right_theta , "float")) 
        
#        ACC_Wrong_to_p2_theta = tf.equal(tf.cast(tf.argmax(y_pred_angle,1),dtype=tf.uint8), 
#                                      tf.cast(tf.zeros_like(Labels_Angle),dtype=tf.uint8))
#        ACC_Wrong_to_p2_theta = tf.reduce_mean(tf.cast(ACC_Wrong_to_p2_theta , "float"))        
#
#        ACC_Wrong_to_p3_theta = tf.equal(tf.cast(tf.argmax(y_pred_angle,1),dtype=tf.uint8), 
#                                      tf.cast(tf.ones_like(Labels_Angle),dtype=tf.uint8))
#        ACC_Wrong_to_p3_theta = tf.reduce_mean(tf.cast(ACC_Wrong_to_p3_theta , "float")) 
        
        return ACC_Right,ACC_Wrong_to_p2,ACC_Wrong_to_p3,ACC_Right_theta, output  
    
    