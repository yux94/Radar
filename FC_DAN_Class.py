import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.contrib.slim as slim

#from tensorflow.contrib.layers.python.layers import batch_norm as bn

#regularizer = tf.contrib.layers.l2_regularizer(scale=0.000c1)#######################
#regularizer = tf.contrib.layers.l1_regularizer(scale=0.0001)#######################

class FC_DAN():
    def __init__(self):
        self.weights=[]
        return    
    def _variable_decay(self,var,wd):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=var.op.name+'_loss')
        tf.add_to_collection('losses', weight_decay)
        return
        
    def _get_fc_weight(self,name,shape,initializer):
#        weight = tf.get_variable(name=name+'_weights',shape=shape,
#                        initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=tf.float32),
#                            dtype=tf.float32)
#                            
#        weight = tf.get_variable(name=name+'_weights', shape=shape,
#                        initializer=tf.random_normal_initializer(stddev=initializer, dtype=tf.float32),
#                            dtype=tf.float32)
        
        weight = tf.get_variable(name=name+'_weights', shape=shape,dtype=tf.float32)# default:glorot_uniform_initializer
        weight = tf.get_variable(name=name+'_weights', shape=shape,dtype=tf.float32,initializer=tf.contirb.layers.xavier_initializer())# default:glorot_uniform_initializer

#        
        return weight                    
    
    
    def _get_conv_filter(self,name,shape,initializer):  
        
#        weight = tf.get_variable(name=name+'_weight',shape=shape,
#                        initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=tf.float32),
#                            dtype=tf.float32)    
        weight = tf.get_variable(name=name+'_weight', shape=shape,
                        initializer=tf.random_normal_initializer(stddev=initializer, dtype=tf.float32),
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
            self._variable_decay(weight,0.0005)
            self._variable_decay(conv_biases,0.0)        
        
        return bias
    
    def _ReLU(self,Bottom):
        relu = tf.nn.relu(Bottom)
        return relu
    
    def Layer_Norm(self,Bottom,eps,name):
        #eps is for math stability
       
        axis = list(range(len(Bottom.get_shape())-1))
        mean,variance = tf.nn.moments(Bottom,axis)  
        gain = tf.get_variable(name=name+'_gain',
                        initializer=tf.ones_initializer(shape=mean.get_shape().as_list(),dtype=tf.float32),
                            dtype=tf.float32)
        bias = tf.get_variable(name=name+'_bias',
                        initializer=tf.zeros_initializer(shape=mean.get_shape().as_list(),dtype=tf.float32),
                            dtype=tf.float32)
        
        ln = (Bottom - mean)/tf.sqrt( variance + eps )
                            
        return ln * gain + bias
    
    def _instance_norm(self,Bottom,train=True):
        batch,rows,cols,channels = [i.value for i in Bottom.get_shape()]
        var_shape = [channels]
        mu , sigma_sq = tf.nn.moments(Bottom,[1,2],keep_dims=True)
        shift = tf.Variable(tf.zeros(var_shape))
        scale = tf.Variable(tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (Bottom - mu)/(sigma_sq + epsilon)**(0.5)
        return scale * normalized + shift
    
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
#             bn = tf.nn.batch_normalization(Bottom,mean,variance,Beta,Gamma,1e-5)
        else:
             bn = tf.nn.batch_normalization(Bottom,moving_mean,moving_variance,Beta,Gamma,1e-5)
#             bn = tf.nn.batch_normalization(Bottom,mean,variance,Beta,Gamma,1e-5)
             
        return bn
        
        
    def _res_unit(self,Bottom,ks,num_out,after_maxpool,name):
        
        if after_maxpool == True:
            resize = self._conv_layer(Bottom,1,num_out,(0.01,0.0),(1,1,1,1),'SAME',name+'_resize')
        else:
            resize = Bottom
        
        bn = self._BatchNormalization(resize,name+'_1')
        relu = self._ReLU(bn)
        conv1 = self._conv_layer(relu,ks,num_out,(0.01,1.0),(1,1,1,1),'SAME',name+'_1')
        bn1 = self._BatchNormalization(conv1,name+'_2')
        relu1  = self._ReLU(bn1)
        conv2 = self._conv_layer(relu1,ks,num_out,(0.01,1.0),(1,1,1,1),'SAME',name+'_2')
        
        res = resize + conv2
        
        return res
    
    def _fully_connect_layer(self,Bottom,output_num,initializer,active,name):
        
        dim = Bottom.get_shape().as_list()[1]
        shape = (dim,output_num)
        weight = self._get_fc_weight(name,shape,initializer[0])
        
#        self.weights.append(weight)####################################
        
        biases = self._get_bias(name,output_num,initializer[1])
        fc = tf.nn.bias_add(tf.matmul(Bottom,weight),biases)

        
        
        if active == 'ReLU':
            fc = tf.nn.relu(fc)
        
        if self.Phase == 'Train':
            self._variable_decay(weight,0.0005)
            self._variable_decay(biases,0.0)        
            
        
        return fc
    
    
    def _Dropout(self,bottom):
        if self.Phase=='Train':
            bottom = tf.nn.dropout(bottom,keep_prob=0.5)
        else:
            bottom = tf.nn.dropout(bottom,keep_prob=1.0)
        return bottom
    
    def _walk_visit_loss(self,bottom,walker_weight=1.0,visit_weight=1.0,reuse=None):#########
        batchsize = bottom.get_shape().as_list()[0] / 2
        a = bottom[0:batchsize,:]#source
        b = bottom[batchsize:,:]#target
        labels = self.labels
        labels = labels[0:batchsize]
        equality_matrix = tf.equal(tf.reshape(labels,[-1,1]),labels)
        equality_matrix = tf.cast(equality_matrix,tf.float32)
        p_target = (equality_matrix/tf.reduce_sum(equality_matrix,[1],keep_dims=True))
        
        match_ab = tf.matmul(a,b,transpose_b=True,name='match_ab')
        p_ab = tf.nn.softmax(match_ab,name='p_ab')
        p_ba = tf.nn.softmax(tf.transpose(match_ab),name='p_ba')
        p_aba = tf.matmul(p_ab,p_ba,name='p_aba')
        
#        estimate_error = self._create_walk_statistics(p_aba,equality_matrix)
        self._create_walk_statistics(p_aba,equality_matrix)
        
        loss_aba = tf.losses.softmax_cross_entropy(p_target,tf.log(1e-8+p_aba),weights=walker_weight,scope='loss_aba')#L_walker
#        visit_loss = self._add_visit_loss(p_ab,visit_weight)#L_visit
        self._add_visit_loss(p_ab,visit_weight)#L_visit
        tf.summary.scalar('Loss_aba',loss_aba)
#        return loss_aba,visit_loss,estimate_error
        
    def _add_visit_loss(self,p,weight=1.0):
        visit_probability = tf.reduce_mean(p,[0],keep_dims=True,name='visit_prob')
        t_nb = tf.shape(p)[1]
        visit_loss = tf.losses.softmax_cross_entropy(tf.fill([1,t_nb],1.0/tf.cast(t_nb,tf.float32)),
                                                     tf.log(1e-8+visit_probability),
                                                     weights=weight,scope='loss_visit')
#        return visit_loss   
        tf.summary.scalar('Loss_Visit',visit_loss)                                       
        
    def _create_walk_statistics(self,p_aba,equality_matrix):
        per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba),1)**0.5
        estimate_error = tf.reduce_mean(1.0-per_row_accuracy,name=p_aba.name[:-2]+'_esterr')
#        self.add_average(estimate_error)
#        self.add_average(p_aba)
        
#        return estimate_error
        tf.summary.scalar('Stats_EstError',estimate_error)
 
    def add_average(self,variable,reuse=None):
        self.step = slim.get_or_create_global_step()
        self.ema = tf.train.ExponentialMovingAverage(0.99,self.step)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,self.ema.apply([variable]))
        average_variable = tf.identity(self.ema.average(variable),name=variable.name[:-2]+'_avg')
        return average_variable

    
    def _MMD_loss(self,bottom,tradeoff):
        batchsize = bottom.get_shape().as_list()[0] / 2
        source_input = bottom[0:batchsize,:]
        target_input = bottom[batchsize:,:]
        
        tf.summary.histogram(bottom.name+'source_feature',source_input)        
        tf.summary.histogram(bottom.name+'target_feature',target_input)    
        
        bandwidth = tf.cast(0,dtype = tf.float32)
        kernel_mul = 2
        for idx in range(batchsize-1):
            square_distance=tf.nn.l2_loss(source_input[idx,:]-target_input[idx,:])*2
            bandwidth+=square_distance
            square_distance=tf.nn.l2_loss(source_input[idx,:]-source_input[idx+1,:])*2
            bandwidth+=square_distance
            square_distance=tf.nn.l2_loss(target_input[idx,:]-target_input[idx+1,:])*2
            bandwidth+=square_distance
        gamma = (batchsize-1)*3 / bandwidth
        times = kernel_mul ** (0.5)
        temp_gamma = gamma / times
        loss = 0
    
        for i in range(batchsize*2):
            temp_gamma = tf.stop_gradient(temp_gamma)
            
            s1=tf.random_uniform(shape=[],minval=0,maxval=batchsize,dtype=tf.int32)
            s2=tf.random_uniform(shape=[],minval=0,maxval=batchsize-1,dtype=tf.int32)
            s2 = tf.cond(tf.equal(s1,s2),lambda:s2+1,lambda:s2)
            
            t1=tf.random_uniform(shape=[],minval=0,maxval=batchsize,dtype=tf.int32)
            t2=tf.random_uniform(shape=[],minval=0,maxval=batchsize-1,dtype=tf.int32)
            t2 = tf.cond(tf.equal(t1,t2),lambda:t2+1,lambda:t2)
            
            loss1 = tf.exp((-temp_gamma*(tf.nn.l2_loss(source_input[s1,:]-source_input[s2,:])*2)))
            loss2 = -tf.exp((-temp_gamma*(tf.nn.l2_loss(source_input[s1,:]-target_input[t2,:])*2)))
            loss3 = -tf.exp((-temp_gamma*(tf.nn.l2_loss(source_input[s2,:]-target_input[t1,:])*2)))
            loss4 = tf.exp((-temp_gamma*(tf.nn.l2_loss(target_input[t1,:]-target_input[t2,:])*2)))
            loss = loss + loss1 + loss2 + loss3 + loss4
        loss = tf.multiply(loss, tf.cast(1.* tradeoff / (batchsize*2) ,dtype=tf.float32), name='mmd_loss')
        tf.add_to_collection('losses',loss)
        
        return loss , gamma
        
    def _reshape_conv_to_fc(self,bottom):
        shape = bottom.get_shape().as_list()
       
        dim = 1
        for d in shape[1:]:
                 dim *= d
        x = tf.reshape(bottom, [-1, dim])
        return x        

    def apply_regularization(self, _lambda):
        # L2 regularization for the fully connected parameters
        regularization = 0.0
        for weights in zip(self.weights):
            regularization += tf.nn.l2_loss(self.weights) 
        # 1e5
        return _lambda * regularization
        
    def Build_FC(self,Bottom,Phase,MMD='UnUse'):
        self.Phase = Phase
        
        
        fc1 = self._fully_connect_layer(Bottom,4096,(0.01,1.0),'ReLU','fc1')
        
        fc2 = self._fully_connect_layer(fc1,4096,(0.01,1.0),'ReLU','fc2')
        
        fc3 = self._fully_connect_layer(fc2,2048,(0.01,1.0),'ReLU','fc3')
        
        fc4 = self._fully_connect_layer(fc3,2048,(0.01,1.0),'ReLU','fc4')
        
        fc5 = self._fully_connect_layer(fc4,1024,(0.01,1.0),'ReLU','fc5')
        
        fc6 = self._fully_connect_layer(fc5,1024,(0.01,1.0),'ReLU','fc6')
        
        if self.Phase=='Train' and MMD=='Use':
            
            MMD1,Gramma1 = self._MMD_loss(fc6,0.01)#1:0.01 2:0.1 3:0.3
        
        fc7 = self._fully_connect_layer(fc6,3,(0.01,0),'None','output')
        
        if self.Phase=='Train' and MMD=='Use':
            MMD2,Gamma2 = self._MMD_loss(fc7,0.3)#1:0.3                                   
            return MMD1,MMD2,fc7
        else:
            return fc7
    
    def Build_CNN(self,Bottom,Phase,MMD='UnUse') :
        self.Phase = Phase
#    def _conv_layer(self,Bottom,ks,num_output,initializer,stride,pad,name):
        batchsize = Bottom.get_shape().as_list()[0]
        dim = Bottom.get_shape().as_list()[1] 
        Bottom = tf.reshape(Bottom,[batchsize,dim,1,1])        
#        if self.Phase == 'Train':
#            training = tf.Variable(True,name='training')
#        else:
#            training = tf.Variable(False,name='testing')
#            
        '''begin convolutional layer'''
        conv1 = self._conv_layer(Bottom,32,128,(0.01,0.0),(1,2,1,1),'SAME','conv1_1')
#        bn1 = tf.cond(training,
#                      lambda:tf.contirb.layers.batch_norm(conv1,activation_fn=tf.nn.relu,is_training=True,reuse=None,name='bn1_1'),
#                      lambda:tf.contirb.layers.batch_norm(conv1,activation_fn=tf.nn.relu,is_training=False,reuse=True,name='bn1_1'))
#        bn1 = tf.layers.batch_normalization(conv1,training=training,name='bn1_1')
#        bn1 = self.Layer_Norm(conv1,1e-5,'ln1_1')
#        relu1  = self._ReLU(conv1)
#        bn1 = self._BatchNormalization(conv1,'bn1_1')
#        bn1 = self._instance_norm(conv1)
#        bn1 = tf.contrib.layers.batch_norm(conv1,scale=True,is_training=training,updates_collections=None,scope='bn1_1')
#        relu1  = self._ReLU(bn1)
        relu1  = self._ReLU(conv1)
        max_pool1 = tf.nn.max_pool(relu1,(1,4,1,1),(1,2,1,1),'VALID')#256
        
        ''' convolutional layer2 '''
        conv2 = self._conv_layer(max_pool1,8,256,(0.01,0.0),(1,1,1,1),'SAME','conv2_1')
#        bn2 = tf.cond(Phase,
#                      lambda:tf.contirb.layers.batch_norm(conv2,activation_fn=tf.nn.relu,is_training=True,reuse=None,name='bn2_1'),
#                      lambda:tf.contirb.layers.batch_norm(conv2,activation_fn=tf.nn.relu,is_training=False,reuse=True,name='bn2_1'))
#        bn2 = tf.layers.batch_normalization(conv2,training=training,name='bn2_1')
#        bn2 = self.Layer_Norm(conv2,1e-5,'ln2_1')
#        relu2  = self._ReLU(conv2)
#        bn2 = self._BatchNormalization(conv2,'bn2_1')
#        bn2 = self._instance_norm(conv2)
#        bn2 = tf.contrib.layers.batch_norm(conv2,scale=True,is_training=training,updates_collections=None,scope='bn2_1')
#        relu2  = self._ReLU(bn2)
        relu2  = self._ReLU(conv2)
        max_pool2 = tf.nn.max_pool(relu2,(1,4,1,1),(1,2,1,1),'VALID')#128

        ''' convolutional layer3 '''
        conv3 = self._conv_layer(max_pool2,8,256,(0.01,0.0),(1,1,1,1),'SAME','conv3_1')
#        bn3 = tf.cond(Phase,
#                      lambda:tf.contirb.layers.batch_norm(conv3,activation_fn=tf.nn.relu,is_training=True,reuse=None,name='bn3_1'),
#                      lambda:tf.contirb.layers.batch_norm(conv3,activation_fn=tf.nn.relu,is_training=False,reuse=True,name='bn3_1'))
#        bn3 = tf.layers.batch_normalization(conv3,training=training,name='bn3_1')
#        bn3 = self.Layer_Norm(conv3,1e-5,'ln3_1')
#        relu3  = self._ReLU(conv3)
#        bn3 = self._BatchNormalization(conv3,'bn3_1')
#        bn3 = self._instance_norm(conv3)
#        bn3 = tf.contrib.layers.batch_norm(conv3,scale=True,is_training=training,updates_collections=None,scope='bn3_1')        
#        relu3  = self._ReLU(bn3)
        relu3  = self._ReLU(conv3)
        max_pool3 = tf.nn.max_pool(relu3,(1,4,1,1),(1,2,1,1),'VALID')#64       

        ''' convolutional layer4 '''
        conv4 = self._conv_layer(max_pool3,4,512,(0.01,0.0),(1,1,1,1),'SAME','conv4_1')
#        bn4 = tf.cond(Phase,
#                      lambda:tf.contirb.layers.batch_norm(conv4,activation_fn=tf.nn.relu,is_training=True,reuse=None,name='bn4_1'),
#                      lambda:tf.contirb.layers.batch_norm(conv4,activation_fn=tf.nn.relu,is_training=False,reuse=True,name='bn4_1'))

#        bn4 = tf.layers.batch_normalization(conv4,training=training,name='bn4_1')
#        bn4 = self.Layer_Norm(conv4,1e-5,'ln4_1')
#        relu4  = self._ReLU(conv4)
#        bn4 = self._BatchNormalization(conv4,'bn4_1')
#        bn4 = self._instance_norm(conv4)
#        bn4 = tf.contrib.layers.batch_norm(conv4,scale=True,is_training=training,updates_collections=None,scope='bn4_1')
#        relu4  = self._ReLU(bn4)
        relu4  = self._ReLU(conv4)        
        max_pool4 = tf.nn.max_pool(relu4,(1,4,1,1),(1,2,1,1),'VALID')#32  

        ''' convolutional layer5 '''
        conv5 = self._conv_layer(max_pool4,4,1024,(0.01,0.0),(1,1,1,1),'SAME','conv5_1')
#        bn5 = tf.cond(Phase,
#                      lambda:tf.contirb.layers.batch_norm(conv5,activation_fn=tf.nn.relu,is_training=True,reuse=None,name='bn5_1'),
#                      lambda:tf.contirb.layers.batch_norm(conv5,activation_fn=tf.nn.relu,is_training=False,reuse=True,name='bn5_1'))

#        bn5 = tf.layers.batch_normalization(conv5,training=training,name='bn5_1')
#        bn5 = self.Layer_Norm(conv5,1e-5,'ln5_1')
#        relu5  = self._ReLU(conv5)
#        bn5 = self._BatchNormalization(conv5,'bn5_1')
#        bn5 = self._instance_norm(conv5)
#        bn5 = tf.contrib.layers.batch_norm(conv5,scale=True,is_training=training,updates_collections=None,scope='bn5_1')
#        relu5  = self._ReLU(bn5)
        relu5  = self._ReLU(conv5)        
        max_pool5 = tf.nn.max_pool(relu5,(1,4,1,1),(1,2,1,1),'VALID')#16  
        
        resize = self._reshape_conv_to_fc(max_pool5)
        
        fc1 = self._fully_connect_layer(resize,4096,(0.005,1.0),'ReLU','fc1')
#        
#        fc1 = self._fully_connect_layer(resize,4096,(0.005,1.0),'None','fc1')
#        fc1 = self._BatchNormalization(fc1,'bnfc_1')
#        fc1 = tf.nn.relu(fc1)

        fc2 = self._fully_connect_layer(fc1,4096,(0.005,1.0),'ReLU','fc2')
#        fc2 = self._fully_connect_layer(fc1,4096,(0.005,1.0),'None','fc2')        
#        fc2 = self._BatchNormalization(fc2,'bnfc_2')
#        fc2 = tf.nn.relu(fc2)

#        fc2 = self._Dropout(fc2)       
        '''Walk visit loss'''        
#        loss_aba,visit_loss,estimate_error = self._walk_visit_loss(fc2)   
#        if self.Phase=='Train' :
#            self._walk_visit_loss(fc2)
            
        if self.Phase=='Train' and MMD=='Use':
            
            tradeoff1 = tf.cond(tf.greater_equal(self.Global_step,1000),
                                lambda:tf.cast(0.01,dtype=tf.float32),lambda:tf.cast(0.0,dtype=tf.float32))
            MMD1,Gramma1 = self._MMD_loss(fc2,tradeoff1)######1:0.01 2:0.05 3:0.1 
        
        fc3 = self._fully_connect_layer(fc2,3,(0.01,0),'None','output')        
        
        if self.Phase=='Train' and MMD=='Use':
            tradeoff2 = tf.cond(tf.greater_equal(self.Global_step,1000),######raw:0.3 4:0.35
                                lambda:tf.cast(0.3,dtype=tf.float32),lambda:tf.cast(0.0,dtype=tf.float32))
            MMD2,Gamma2 = self._MMD_loss(fc3,tradeoff2)                                          
            return MMD1,MMD2,fc3
        else:
            return fc3   

    def Build_ResNet(self,Bottom,Phase,MMD='UnUse'):
        self.Phase = Phase
        
        batchsize = Bottom.get_shape().as_list()[0]
        dim = Bottom.get_shape().as_list()[1] 
        Bottom = tf.reshape(Bottom,[batchsize,dim,1,1])
        
        '''begin convolutional layer'''
        conv1 = self._conv_layer(Bottom,128,128,(0.01,0.0),(1,2,1,1),'SAME','conv1')
#        bn1 = self._BatchNormalization(conv1,'bn1')
#        relu1  = self._ReLU(bn1)
        relu1  = self._ReLU(conv1)

        max_pool1 = tf.nn.max_pool(relu1,(1,16,1,1),(1,2,1,1),'SAME')#256
        
        '''256 dim'''
        res_1 = self._res_unit(max_pool1,32,256,True,'Res1')
        res_2 = self._res_unit(res_1,32,256,False,'Res2')
        max_pool2 = tf.nn.max_pool(res_2,(1,5,1,1),(1,2,1,1),'SAME')#128
        '''128 dim'''
        res_3 = self._res_unit(max_pool2,16,512,True,'Res3')
        res_4 = self._res_unit(res_3,16,512,False,'Res4')
        max_pool3 = tf.nn.max_pool(res_4,(1,5,1,1),(1,2,1,1),'SAME')#64
        '''64 dim'''
        res_5 = self._res_unit(max_pool3,8,512,True,'Res5')
        res_6 = self._res_unit(res_5,8,512,False,'Res6')
        max_pool4 = tf.nn.max_pool(res_6,(1,5,1,1),(1,2,1,1),'SAME')#32
        '''32 dim'''
        res_7 = self._res_unit(max_pool4,8,1024,True,'Res7')
        res_8 = self._res_unit(res_7,8,1024,False,'Res8')
        max_pool5 = tf.nn.max_pool(res_8,(1,5,1,1),(1,2,1,1),'SAME')#16
        
        resize = self._reshape_conv_to_fc(max_pool5)
        fc1 = self._fully_connect_layer(resize,4096,(0.005,1.0),'ReLU','fc1')
        fc2 = self._fully_connect_layer(fc1,4096,(0.005,1.0),'ReLU','fc2')
        fc2 = self._Dropout(fc2)    ############   
       
        
        if self.Phase=='Train' and MMD=='Use':
            
            MMD1,Gramma1 = self._MMD_loss(fc2,0.3)
        
        fc3 = self._fully_connect_layer(fc2,3,(0.01,0),'None','output')
        
        if self.Phase=='Train' and MMD=='Use':
            MMD2,Gamma2 = self._MMD_loss(fc3,0.3)                                          
            return MMD1,MMD2,fc3
        else:
            return fc3                  

    def Loss(self,Bottom,Labels,Phase,MMD='UnUse'):                                           
        
        labels = tf.cast(Labels, tf.int64)
        self.labels=labels
        
        if  MMD=='Use': 
            MMD1,MMD2,output = self.Build_CNN(Bottom,Phase,MMD) 
#            MMD1,MMD2,output = self.Build_ResNet(Bottom,Phase,MMD)
        else:
            output = self.Build_CNN(Bottom,Phase,MMD) 


#            output = self.Build_ResNet(Bottom,Phase,MMD)                               
        
#        tf.add_to_collection('losses', loss_aba)
#        tf.add_to_collection('losses', visit_loss)

#        logit_loss=tf.losses.softmax_cross_entropy(tf.one_hot(labels,output.get_shape()[-1]),
#                                                   output,scope='loss_logit',weights=1.0,label_smoothing=0.0)
#        tf.summary.scalar('Loss_Logit',logit_loss)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=output)                          
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)#####################
#        reg_loss = tf.contrib.layers.apply_regularization(regularizer, reg_variables)####################
#        loss_all = loss_aba + visit_loss + cross_entropy_mean
        tf.add_to_collection('losses', cross_entropy_mean)
#        tf.add_to_collection('losses', loss_all)
#        total_loss = tf.losses.get_total_loss()
#        total_loss_average = self.add_average(total_loss)
        total_loss = tf.add_n(tf.get_collection('losses'), name='Label_Loss')  
 
#        
        if self.Phase=='Train' and MMD=='Use':
            return total_loss,MMD1,MMD2
        else:
            return total_loss
            
    def Train(self,Bottom,Labels,base_lr,Global_step,MMD ='UnUse'):    
        
        self.Global_step = Global_step
        if MMD == 'Use':
            total_loss,MMD1,MMD2 = self.Loss(Bottom,Labels,Phase='Train',MMD=MMD)  
        else:
            total_loss = self.Loss(Bottom,Labels,Phase='Train',MMD=MMD)                                     
        
        lr = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step,decay_steps=50000,decay_rate=0.1,staircase=True)
#        opt = tf.train.GradientDescentOptimizer(lr)
#        opt=tf.train.MomentumOptimizer(lr, momentum=0.9)                                            
        opt=tf.train.AdamOptimizer(lr)
        Grad = opt.compute_gradients(total_loss)                                          
        train_op = opt.apply_gradients(Grad,global_step=Global_step)
#        train_op = slim.learning.create_train_op(total_loss,opt)

        if MMD == 'Use':
            return train_op,total_loss,MMD1,MMD2  
        else:
            return train_op,total_loss                                       
    
    def Accuracy(self,Bottom,Labels):
        
        output = self.Build_CNN(Bottom,Phase='Test')
#        output = self.Build_ResNet(Bottom,Phase='Test')
        ACC = tf.equal(tf.cast(tf.argmax(output,1),dtype=tf.uint8), 
                                      tf.cast(Labels,dtype=tf.uint8))
        ACC = tf.reduce_mean(tf.cast(ACC, "float")) 
        
        return output,ACC    
        
    
#angles = tf.concat([Train_angle3,Test_angle3],0)                                          
#Batch1 = tf.concat([Test_Batch1,Test_Batch2,Train_Batch3,Test_Batch3],0)#,Train_Batch2,Test_Batch2,Train_Batch3,Test_Batch3],0)
#Labels1 = tf.concat([Test_Labels1,Test_Labels2,Train_Labels3,Test_Labels3],0)#,Train_Labels2,Test_Labels2,Train_Labels3,Test_Labels3],0)
#ACC_Right2,ACC2_Wrong_to_p5,ACC2_Wrong_to_p6,P2_output,ACC_Right5,ACC5_Wrong_to_p2,
#ACC5_Wrong_to_p6,P5_output,ACC_Right,ACC_Wrong_to_p2,ACC_Wrong_to_p3,P6_output=Transfer.Detail_P6_ACC(Batch1,Labels1)

    def Detail_P6_ACC(self,Bottom,Labels,angles):
        
#        '''angle detail information'''
#        p2_angles = np.zeros(24) 
#        p5_angles = np.zeros(24)
#        p6_angles = np.zeros(24)
        
#        labels = tf.cast(Labels, tf.int64)
        output = self.Build_CNN(Bottom,Phase='Test')
#        loss_aba,visit_loss,output = self.Build_CNN(Bottom,Phase='Test')
        #Source:0,Target:1
        Size = output.get_shape().as_list()[0] * 1 / 4
        P2_output = output[0:Size*1,:]
        P5_output = output[Size*1:Size*2,:]
        P6_output = output[Size*2:,:]
        Label2 = Labels[0:Size]
        Label5 = Labels[Size:Size*2]
        Label6 = Labels[Size*2:]
        angles = angles
        y_true2 = [2 for i in range(np.size(Label2))]#P2 to p6
        y_true5 = [2 for i in range(np.size(Label5))]#p5 to p6

#        output = self.Build_ResNet(Bottom,Phase='Test')
        ACC_Right2 = tf.equal(tf.cast(tf.argmax(P2_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.zeros_like(Label2),dtype=tf.uint8))
        ACC_Right2 = tf.reduce_mean(tf.cast(ACC_Right2, "float")) 
        
        ACC2_Wrong_to_p5 = tf.equal(tf.cast(tf.argmax(P2_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.ones_like(Label2),dtype=tf.uint8))
        ACC2_Wrong_to_p5 = tf.reduce_mean(tf.cast(ACC2_Wrong_to_p5, "float"))        

        ACC2_Wrong_to_p6 = tf.equal(tf.cast(tf.argmax(P2_output,1),dtype=tf.uint8), 
                                      tf.cast(y_true2,dtype=tf.uint8))
        ACC2_Wrong_to_p6 = tf.reduce_mean(tf.cast(ACC2_Wrong_to_p6, "float"))  
        
        #        output = self.Build_ResNet(Bottom,Phase='Test')
        ACC_Right5 = tf.equal(tf.cast(tf.argmax(P5_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.ones_like(Label5),dtype=tf.uint8))
        ACC_Right5 = tf.reduce_mean(tf.cast(ACC_Right5, "float")) 
        
        ACC5_Wrong_to_p2 = tf.equal(tf.cast(tf.argmax(P5_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.zeros_like(Label5),dtype=tf.uint8))
        ACC5_Wrong_to_p2 = tf.reduce_mean(tf.cast(ACC5_Wrong_to_p2, "float"))        

        ACC5_Wrong_to_p6 = tf.equal(tf.cast(tf.argmax(P5_output,1),dtype=tf.uint8), 
                                      tf.cast(y_true5,dtype=tf.uint8))
        ACC5_Wrong_to_p6 = tf.reduce_mean(tf.cast(ACC5_Wrong_to_p6, "float"))  
        
#        output = self.Build_ResNet(Bottom,Phase='Test')
        ACC_Right = tf.equal(tf.cast(tf.argmax(P6_output,1),dtype=tf.uint8), 
                                      tf.cast(Label6,dtype=tf.uint8))
        ACC_Right = tf.reduce_mean(tf.cast(ACC_Right, "float")) 
        
        ACC_Wrong_to_p2 = tf.equal(tf.cast(tf.argmax(P6_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.zeros_like(Label6),dtype=tf.uint8))
        ACC_Wrong_to_p2 = tf.reduce_mean(tf.cast(ACC_Wrong_to_p2, "float"))        
        
        ACC_Wrong_to_p3 = tf.equal(tf.cast(tf.argmax(P6_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.ones_like(Label6),dtype=tf.uint8))
        ACC_Wrong_to_p3 = tf.reduce_mean(tf.cast(ACC_Wrong_to_p3, "float")) 
        
#        p6_output = np.argmax(P6_output,axis=-1)
#        for item in range(p6_output.size):
#            if p6_output[item] == 0 :
#                '''be mistaken for p2'''
#                angle_label = angles[item]
#                p2_angles[angle_label] += 1
                
#            elif p6_output[item] == 1 :
#                '''be mistaken for p5'''
#                angle_label = angles[item]
#                p5_angles[angle_label] += 1
            
#            elif p6_output[item] ==2 :
#                '''correctly classified to p6'''
#                angle_label = angles[item]
#                p6_angles[angle_label] += 1

         
        '''correct angle'''
#        p6_angles
        '''p2 angle'''
#        p2_angles
        '''p5 angle'''
#        p5_angles        
        

        return ACC_Right2,ACC2_Wrong_to_p5,ACC2_Wrong_to_p6,P2_output,ACC_Right5,ACC5_Wrong_to_p2,ACC5_Wrong_to_p6,P5_output,ACC_Right,ACC_Wrong_to_p2,ACC_Wrong_to_p3,P6_output
    
    
    def Detail_P5_P6_ACC(self,Bottom,Labels,angles):
        
#        '''angle detail information'''
#        p2_angles = np.zeros(24) 
#        p5_angles = np.zeros(24)
#        p6_angles = np.zeros(24)
        
#        labels = tf.cast(Labels, tf.int64)
        output = self.Build_CNN(Bottom,Phase='Test')
#        loss_aba,visit_loss,output = self.Build_CNN(Bottom,Phase='Test')
        #Source:0,Target:1
        Size = output.get_shape().as_list()[0] * 1 / 5
        P2_output = output[0:Size*1,:]
        P5_output = output[Size*1:Size*3,:]
        P6_output = output[Size*3:,:]
        Label2 = Labels[0:Size]
        Label5 = Labels[Size:Size*3]
        Label6 = Labels[Size*3:]
        angles = angles
        y_true2 = [2 for i in range(np.size(Label2))]#P2 to p6
        y_true5 = [2 for i in range(np.size(Label5))]#p5 to p6

#        output = self.Build_ResNet(Bottom,Phase='Test')
        ACC_Right2 = tf.equal(tf.cast(tf.argmax(P2_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.zeros_like(Label2),dtype=tf.uint8))
        ACC_Right2 = tf.reduce_mean(tf.cast(ACC_Right2, "float")) 
        
        ACC2_Wrong_to_p5 = tf.equal(tf.cast(tf.argmax(P2_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.ones_like(Label2),dtype=tf.uint8))
        ACC2_Wrong_to_p5 = tf.reduce_mean(tf.cast(ACC2_Wrong_to_p5, "float"))        

        ACC2_Wrong_to_p6 = tf.equal(tf.cast(tf.argmax(P2_output,1),dtype=tf.uint8), 
                                      tf.cast(y_true2,dtype=tf.uint8))
        ACC2_Wrong_to_p6 = tf.reduce_mean(tf.cast(ACC2_Wrong_to_p6, "float"))  
        
        #        output = self.Build_ResNet(Bottom,Phase='Test')
        ACC_Right5 = tf.equal(tf.cast(tf.argmax(P5_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.ones_like(Label5),dtype=tf.uint8))
        ACC_Right5 = tf.reduce_mean(tf.cast(ACC_Right5, "float")) 
        
        ACC5_Wrong_to_p2 = tf.equal(tf.cast(tf.argmax(P5_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.zeros_like(Label5),dtype=tf.uint8))
        ACC5_Wrong_to_p2 = tf.reduce_mean(tf.cast(ACC5_Wrong_to_p2, "float"))        

        ACC5_Wrong_to_p6 = tf.equal(tf.cast(tf.argmax(P5_output,1),dtype=tf.uint8), 
                                      tf.cast(y_true5,dtype=tf.uint8))
        ACC5_Wrong_to_p6 = tf.reduce_mean(tf.cast(ACC5_Wrong_to_p6, "float"))  
        
#        output = self.Build_ResNet(Bottom,Phase='Test')
        ACC_Right = tf.equal(tf.cast(tf.argmax(P6_output,1),dtype=tf.uint8), 
                                      tf.cast(Label6,dtype=tf.uint8))
        ACC_Right = tf.reduce_mean(tf.cast(ACC_Right, "float")) 
        
        ACC_Wrong_to_p2 = tf.equal(tf.cast(tf.argmax(P6_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.zeros_like(Label6),dtype=tf.uint8))
        ACC_Wrong_to_p2 = tf.reduce_mean(tf.cast(ACC_Wrong_to_p2, "float"))        
        
        ACC_Wrong_to_p3 = tf.equal(tf.cast(tf.argmax(P6_output,1),dtype=tf.uint8), 
                                      tf.cast(tf.ones_like(Label6),dtype=tf.uint8))
        ACC_Wrong_to_p3 = tf.reduce_mean(tf.cast(ACC_Wrong_to_p3, "float")) 
        
#        p6_output = np.argmax(P6_output,axis=-1)
#        for item in range(p6_output.size):
#            if p6_output[item] == 0 :
#                '''be mistaken for p2'''
#                angle_label = angles[item]
#                p2_angles[angle_label] += 1
                
#            elif p6_output[item] == 1 :
#                '''be mistaken for p5'''
#                angle_label = angles[item]
#                p5_angles[angle_label] += 1
            
#            elif p6_output[item] ==2 :
#                '''correctly classified to p6'''
#                angle_label = angles[item]
#                p6_angles[angle_label] += 1

         
        '''correct angle'''
#        p6_angles
        '''p2 angle'''
#        p2_angles
        '''p5 angle'''
#        p5_angles        
        

        return ACC_Right2,ACC2_Wrong_to_p5,ACC2_Wrong_to_p6,P2_output,ACC_Right5,ACC5_Wrong_to_p2,ACC5_Wrong_to_p6,P5_output,ACC_Right,ACC_Wrong_to_p2,ACC_Wrong_to_p3,P6_output
    
    