# -*- coding: utf-8 -*-
import tensorflow as tf

class Adversal_discriminative():
    
    def __init__(self):
        return  
    
    def _variable_decay(self,var,wd):
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=var.op.name+'_loss')
        tf.add_to_collection('losses', weight_decay)
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
            self._variable_decay(weight,0.0005)
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

    def _fully_connect_layer(self,Bottom,output_num,initializer,active,name):
        
        dim = Bottom.get_shape().as_list()[1]
        shape = (dim,output_num)
        weight = self._get_fc_weight(name,shape,initializer[0])
        biases = self._get_bias(name,output_num,initializer[1])
        fc = tf.nn.bias_add(tf.matmul(Bottom,weight),biases)
        
        if active == 'ReLU':
            fc = tf.nn.relu(fc)
        
        if self.Phase == 'Train':
            self._variable_decay(weight,0.0005)
            self._variable_decay(biases,0.0)        
        
        return fc
        
    def _reshape_conv_to_fc(self,bottom):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
                 dim *= d
        x = tf.reshape(bottom, [-1, dim])
        return x    
   
  
    def Build_CNN_SOURCE(self,Bottom,Phase,reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        self.Phase = Phase
        
        batchsize = Bottom.get_shape().as_list()[0]
        dim = Bottom.get_shape().as_list()[1] 
        Bottom = tf.reshape(Bottom,[batchsize,dim,1,1])        

        '''begin convolutional layer'''
        conv1 = self._conv_layer(Bottom,32,128,(0.01,0.0),(1,2,1,1),'SAME','sour_conv1')
#        bn1 = self._BatchNormalization(conv1,'bn1_1')
#        relu1  = self._ReLU(bn1)
        relu1  = self._ReLU(conv1)
        
        max_pool1 = tf.nn.max_pool(relu1,(1,4,1,1),(1,2,1,1),'VALID')#256
        
        ''' convolutional layer2 '''
        conv2 = self._conv_layer(max_pool1,8,256,(0.01,0.0),(1,1,1,1),'SAME','sour_conv2')
#        bn2 = self._BatchNormalization(conv2,'bn2_1')
#        relu2  = self._ReLU(bn2)
        relu2  = self._ReLU(conv2)

        max_pool2 = tf.nn.max_pool(relu2,(1,4,1,1),(1,2,1,1),'VALID')#128

        ''' convolutional layer3 '''
        conv3 = self._conv_layer(max_pool2,8,256,(0.01,0.0),(1,1,1,1),'SAME','sour_conv3')
#        bn3 = self._BatchNormalization(conv3,'bn3_1')
#        relu3  = self._ReLU(bn3)
        relu3  = self._ReLU(conv3)
        
        max_pool3 = tf.nn.max_pool(relu3,(1,4,1,1),(1,2,1,1),'VALID')#64       

        ''' convolutional layer4 '''
        conv4 = self._conv_layer(max_pool3,4,512,(0.01,0.0),(1,1,1,1),'SAME','sour_conv4')
#        bn4 = self._BatchNormalization(conv4,'bn4_1')
#        relu4  = self._ReLU(bn4)
        relu4  = self._ReLU(conv4)
        
        max_pool4 = tf.nn.max_pool(relu4,(1,4,1,1),(1,2,1,1),'VALID')#32  

        ''' convolutional layer5 '''
        conv5 = self._conv_layer(max_pool4,4,1024,(0.01,0.0),(1,1,1,1),'SAME','sour_conv5')
#        bn5 = self._BatchNormalization(conv5,'bn5_1')
#        relu5  = self._ReLU(bn5)
        relu5  = self._ReLU(conv5)
        
        
        max_pool5 = tf.nn.max_pool(relu5,(1,4,1,1),(1,2,1,1),'VALID')#16  
        
        resize = self._reshape_conv_to_fc(max_pool5)
        
        fc1 = self._fully_connect_layer(resize,4096,(0.005,1.0),'ReLU','sour_fc1')
        fc2 = self._fully_connect_layer(fc1,4096,(0.005,1.0),'ReLU','sour_fc2')

        return fc2 
            
    def classifier(self,Bottom,reuse=False):
       if reuse:
           tf.get_variable_scope().reuse_variables()
            
       '''Classifier'''
       fc3 = self._fully_connect_layer(Bottom,3,(0.01,0),'None','output')
       
       return fc3         
    
    def discrimination(self,Bottom,reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        ad1 = self._fully_connect_layer(Bottom,1024,(0.01,1.0),'ReLU','adversal_1')
        ad2 = self._fully_connect_layer(ad1,1024,(0.01,1.0),'ReLU','adversal_2')
        domain_output = self._fully_connect_layer(ad2,2,(0.01,0.1),'None','adversal_3')
        
        return tf.nn.sigmoid(domain_output),domain_output
        
    def Build_CNN_TARGET(self,Bottom,Phase,reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        self.Phase = Phase
        
        batchsize = Bottom.get_shape().as_list()[0]
        dim = Bottom.get_shape().as_list()[1] 
        Bottom = tf.reshape(Bottom,[batchsize,dim,1,1])        

        '''begin convolutional layer'''
        conv1 = self._conv_layer(Bottom,32,128,(0.01,0.0),(1,2,1,1),'SAME','tar_conv1')
#        bn1 = self._BatchNormalization(conv1,'bn1_1')
#        relu1  = self._ReLU(bn1)
        relu1  = self._ReLU(conv1)
        
        max_pool1 = tf.nn.max_pool(relu1,(1,4,1,1),(1,2,1,1),'VALID')#256
        
        ''' convolutional layer2 '''
        conv2 = self._conv_layer(max_pool1,8,256,(0.01,0.0),(1,1,1,1),'SAME','tar_conv2')
#        bn2 = self._BatchNormalization(conv2,'bn2_1')
#        relu2  = self._ReLU(bn2)
        relu2  = self._ReLU(conv2)

        max_pool2 = tf.nn.max_pool(relu2,(1,4,1,1),(1,2,1,1),'VALID')#128

        ''' convolutional layer3 '''
        conv3 = self._conv_layer(max_pool2,8,256,(0.01,0.0),(1,1,1,1),'SAME','tar_conv3')
#        bn3 = self._BatchNormalization(conv3,'bn3_1')
#        relu3  = self._ReLU(bn3)
        relu3  = self._ReLU(conv3)
        
        max_pool3 = tf.nn.max_pool(relu3,(1,4,1,1),(1,2,1,1),'VALID')#64       

        ''' convolutional layer4 '''
        conv4 = self._conv_layer(max_pool3,4,512,(0.01,0.0),(1,1,1,1),'SAME','tar_conv4')
#        bn4 = self._BatchNormalization(conv4,'bn4_1')
#        relu4  = self._ReLU(bn4)
        relu4  = self._ReLU(conv4)
        
        max_pool4 = tf.nn.max_pool(relu4,(1,4,1,1),(1,2,1,1),'VALID')#32  

        ''' convolutional layer5 '''
        conv5 = self._conv_layer(max_pool4,4,1024,(0.01,0.0),(1,1,1,1),'SAME','tar_conv5')
#        bn5 = self._BatchNormalization(conv5,'bn5_1')
#        relu5  = self._ReLU(bn5)
        relu5  = self._ReLU(conv5)
        
        
        max_pool5 = tf.nn.max_pool(relu5,(1,4,1,1),(1,2,1,1),'VALID')#16  
        
        resize = self._reshape_conv_to_fc(max_pool5)
        
        fc1 = self._fully_connect_layer(resize,4096,(0.005,1.0),'ReLU','tar_fc1')
        fc2 = self._fully_connect_layer(fc1,4096,(0.005,1.0),'ReLU','tar_fc2')
        
        return fc2 
                        
            
    def Loss(self,Bottom,Labels,Domain): 
        
        if Domain == 'source':
            self.M_X_S= self.Build_CNN_SOURCE(Bottom,'train')
            self.Label_output_source = self.classifier(self.M_X_S)
            self.labels_source = tf.cast(Labels, tf.int64)
        
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.Label_output_source,self.labels_source)                          
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)
            total_loss = tf.add_n(tf.get_collection('losses'), name='Label_Loss')

            return total_loss
        
        else:
            
            M_X_T = self.Build_CNN_TARGET(Bottom,'train')
#            Label_output_target = self.classifier(M_X_T)
#            labels_target = tf.cast(Labels, tf.int64)
            ds_,dis_source = self.discrimination(self.M_X_S)
            dt_,dis_target = self.discrimination(M_X_T,reuse=True)
            #Source:1,Target:0
    
            d_loss_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(dis_source, tf.ones_like(ds_)))
            d_loss_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(dis_target, tf.zeros_like(dt_)))       
            
            d_loss = d_loss_source + d_loss_target
            tf.add_to_collection('losses', d_loss)
            total_loss = tf.add_n(tf.get_collection('losses'), name='Domain_Loss')
        
            return total_loss   
        

    def Pre_training(self,Bottom,Labels,base_lr,Global_step):    
        
        total_loss = self.Loss(Bottom,Labels,Domain='source')                               
        
        lr = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step,decay_steps=10000,decay_rate=0.5,staircase=True)
        
        t_vars = tf.trainable_variables()
        source_var = [var for var in t_vars if 'sour' in var.name]

        opt=tf.train.AdamOptimizer(lr)
        Common_Grad = opt.compute_gradients(total_loss,var_list=source_var)
        Train_op = opt.apply_gradients(Common_Grad,global_step=Global_step)###############################  
        
        return Train_op,total_loss                     
    
    def Adversarial_adaptation(self,Bottom,Labels,base_lr,Global_step):    
                
        total_loss = self.Loss(Bottom,Labels,Domain='target')                               
        
        lr = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step,decay_steps=10000,decay_rate=0.5,staircase=True)
        
        t_vars = tf.trainable_variables()
        target_var = [var for var in t_vars if 'tar' in var.name]                                 
        
        opt=tf.train.AdamOptimizer(lr)
        Common_Grad = opt.compute_gradients(total_loss,var_list=target_var)    
        Train_op = opt.apply_gradients(Common_Grad,global_step=Global_step) 
        
        return Train_op,total_loss     

          
    
    def Accuracy(self,Bottom,Labels,Domain):
                
        output = self.Build_CNN_TARGET(Bottom,Phase='Test',reuse=True)
        output = self.classifier(output,reuse=True)
        
        ACC = tf.equal(tf.cast(tf.argmax(output,1),dtype=tf.uint8), 
                                      tf.cast(Labels,dtype=tf.uint8))
        ACC = tf.reduce_mean(tf.cast(ACC, "float")) 
        
        return output,ACC    
