import tensorflow as tf
from functools import partial
import utils

regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0001)
#from tensorflow.contrib.layers.python.layers import batch_norm as bn
class FC_DAN():
    def __init__(self):
        return    
    def _variable_decay(self,var,wd):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=var.op.name+'_loss')
        tf.add_to_collection('losses', weight_decay)
        return
        
    def _get_fc_weight(self,name,shape,initializer):
        weight = tf.get_variable(name=name+'_weights',shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=tf.float32),
                            dtype=tf.float32)
        
        
        ##########L1 L2 regularization
#        weight = tf.get_variable(name=name+'_weights',regularizer = regularizer, shape=shape,
#                        initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=tf.float32),
#                            dtype=tf.float32)
        
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
    
    def parametric_relu(self,Bottom,name):
        alphas = tf.get_variable(name+'_alpha',Bottom.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(Bottom)
        neg = alphas * (Bottom - abs(Bottom)) * 0.5
        
        return pos + neg         
    
    def _ReLU(self,Bottom):
        relu = tf.nn.relu(Bottom)
        return relu
    
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
             bn = tf.nn.batch_normalization(Bottom,moving_mean,moving_variance,Beta,Gamma,1e-3)#1e-5
#             bn = tf.nn.batch_normalization(Bottom,mean,variance,Beta,Gamma,1e-3)

        else:
             bn = tf.nn.batch_normalization(Bottom,moving_mean,moving_variance,Beta,Gamma,1e-3)#0.001
             
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
    
    def _softmax(self,Bottom,output_num,initializer,active,name):
        dim = Bottom.get_shape().as_list()[1]
        shape = (dim,output_num)
        weight = self._get_fc_weight(name,shape,initializer[0])
        biases = self._get_bias(name,output_num,initializer[1])
        fc = tf.nn.softmax(tf.matmul(Bottom,weight)+biases)
        
        if self.Phase == 'Train':
            self._variable_decay(weight,0.0005)
            self._variable_decay(biases,0.0)        
        
        return fc
    
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
    
    
    def _Dropout(self,bottom):
        if self.Phase=='Train':
            bottom = tf.nn.dropout(bottom,keep_prob=0.5)
        else:
            bottom = tf.nn.dropout(bottom,keep_prob=1.0)
        return bottom
    
    ####tensorlow domain separation MMD########################
    
    def maximum_mean_discrepancy(self,x,y,kernel=utils.gaussian_kernel_matrix):
        with tf.name_scope('MaximumMeanDiscrepancy'):
            cost = tf.reduce_mean(kernel(x,y))
            cost += tf.reduce_mean(kernel(x,y))
            cost -= 2 * tf.reduce_mean(kernel(x,y))
            
            cost = tf.where(cost>0,cost,0,name='value')
        return cost   
    
    def MMD_loss(self,bottom,tradeoff):
        batchsize = bottom.get_shape().as_list()[0] / 2
        source_input = bottom[0:batchsize,:]
        target_input = bottom[batchsize:,:]

        sigmas=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,15,20,25,30,35,100,1e3,1e4,1e5,1e6]
        gaussian_kernel = partial(utils.gaussian_kernel_matrix,sigmas=tf.constant(sigmas))
        loss_value = self.maximum_mean_discrepancy(source_input,target_input,kernel=gaussian_kernel)
        loss_value = tf.maximum(1e-4,loss_value)* tradeoff
        assert_op = tf.Assert(tf.is_finite(loss_value),[loss_value])
        with tf.control_dependencies([assert_op]):
            tag = 'MMD_Loss'
            tf.summary.scalar(tag,loss_value)
            tf.losses.add_loss(loss_value)
        return loss_value
        
    ############end#########################
    
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
        
        batchsize = Bottom.get_shape().as_list()[0]
        dim = Bottom.get_shape().as_list()[1] 
        Bottom = tf.reshape(Bottom,[batchsize,dim,1,1])        

        if self.Phase =='Train':
            training = tf.Variable(True,name='training')
        else:
            training = tf.Variable(False,name='testing')

        '''begin convolutional layer'''
        conv1 = self._conv_layer(Bottom,32,128,(0.01,0.0),(1,2,1,1),'SAME','conv1_1')
#        bn1 = self._BatchNormalization(conv1,'bn1_1')
#        relu1  = self.parametric_relu(conv1,'prelu1') 
#        bn1 = self._BatchNormalization(relu1,'bn1_1')
#        bn1 = tf.layers.batch_normalization(relu1,training=training,name='bn1_1')
#        relu1  = self._ReLU(bn1)
        relu1  = self._ReLU(conv1)       
        max_pool1 = tf.nn.max_pool(relu1,(1,4,1,1),(1,2,1,1),'VALID')#256
#        max_pool1 = tf.nn.local_response_normalization(max_pool1, 2.5, 2, 1e-4, 0.75) ####################
        
        
        ''' convolutional layer2 '''
        conv2 = self._conv_layer(max_pool1,8,256,(0.01,0.0),(1,1,1,1),'SAME','conv2_1')
#        bn2 = self._BatchNormalization(conv2,'bn2_1')
#        relu2  = self.parametric_relu(conv2,'prelu2')
#        bn2 = self._BatchNormalization(relu2,'bn2_1')
#        bn2 = tf.layers.batch_normalization(relu2,training=training,name='bn2_1')
#        relu2  = self._ReLU(bn2)
        relu2  = self._ReLU(conv2)
        max_pool2 = tf.nn.max_pool(relu2,(1,4,1,1),(1,2,1,1),'VALID')#128
#        max_pool2 = tf.nn.local_response_normalization(max_pool2,2.5,2,1e-4,0.75) #########################
        

        ''' convolutional layer3 '''
        conv3 = self._conv_layer(max_pool2,8,256,(0.01,0.0),(1,1,1,1),'SAME','conv3_1')
#        bn3 = self._BatchNormalization(conv3,'bn3_1')
#        relu3  = self.parametric_relu(conv3,'prelu3')
#        bn3 = self._BatchNormalization(relu3,'bn3_1')
#        bn3 = tf.layers.batch_normalization(relu3,training=training,name='bn3_1')
#        relu3  = self._ReLU(bn3)
        relu3  = self._ReLU(conv3)
        max_pool3 = tf.nn.max_pool(relu3,(1,4,1,1),(1,2,1,1),'VALID')#64  
#        max_pool3 = tf.nn.local_response_normalization(max_pool3,2.5,2,1e-4,0.75) ########################      
        
             

        ''' convolutional layer4 '''
        conv4 = self._conv_layer(max_pool3,4,512,(0.01,0.0),(1,1,1,1),'SAME','conv4_1')
#        bn4 = self._BatchNormalization(conv4,'bn4_1')
#        relu4  = self.parametric_relu(conv4,'prelu4')
#        bn4 = self._BatchNormalization(relu4,'bn4_1')
#        bn4 = tf.layers.batch_normalization(relu4,training=training,name='bn4_1')
#        relu4  = self._ReLU(bn4)
        relu4  = self._ReLU(conv4)
        max_pool4 = tf.nn.max_pool(relu4,(1,4,1,1),(1,2,1,1),'VALID')#32  
#        max_pool4 = tf.nn.local_response_normalization(max_pool4,2.5,2,1e-4,0.75) ########################
        
        ''' convolutional layer5 '''
        conv5 = self._conv_layer(max_pool4,4,1024,(0.01,0.0),(1,1,1,1),'SAME','conv5_1')
#        bn5 = self._BatchNormalization(conv5,'bn5_1')
#        relu5  = self.parametric_relu(conv5,'prelu5')
#        bn5 = self._BatchNormalization(relu5,'bn5_1')
#        bn5 = tf.layers.batch_normalization(relu5,training=training,name='bn5_1')
#        relu5  = self._ReLU(bn5)
        relu5  = self._ReLU(conv5)                      
        max_pool5 = tf.nn.max_pool(relu5,(1,4,1,1),(1,2,1,1),'VALID')#16  
#        max_pool5 = tf.nn.local_response_normalization(max_pool5,depth_radius=2.5,bias=2,alpha=1e-4,beta=0.75) ########################
        
        resize = self._reshape_conv_to_fc(max_pool5)
        
        fc1 = self._fully_connect_layer(resize,4096,(0.005,1.0),'ReLU','fc1')
        fc2 = self._fully_connect_layer(fc1,4096,(0.005,1.0),'ReLU','fc2')
#        fc2 = self._Dropout(fc2)
        
        if self.Phase=='Train' and MMD=='Use':
            
            tradeoff1 = tf.cond(tf.greater_equal(self.Global_step,1000),
                                lambda:tf.cast(0.1,dtype=tf.float32),lambda:tf.cast(0.0,dtype=tf.float32))
            MMD1,Gramma1 = self._MMD_loss(fc2,tradeoff1)######1:0.01 2:0.05 3:0.1 
#            MMD1 = self.MMD_loss(fc2,tradeoff1)#############new mmd
        
        fc3 = self._fully_connect_layer(fc2,3,(0.01,0),'None','output')
#        fc3 = self._softmax(fc2,3,(0.01,0),'None','output')

        if self.Phase=='Train' and MMD=='Use':
            tradeoff2 = tf.cond(tf.greater_equal(self.Global_step,1000),######1:0.3
                                lambda:tf.cast(0.3,dtype=tf.float32),lambda:tf.cast(0.0,dtype=tf.float32))
            MMD2,Gamma2 = self._MMD_loss(fc3,tradeoff2)    
#            MMD2 = self.MMD_loss(fc3,tradeoff2)
                                      
            return MMD1,MMD2,fc3
        else:
            return fc3        

#    def Build_ResNet(self,Bottom,Phase,MMD='UnUse'):
#        self.Phase = Phase
#        
#        batchsize = Bottom.get_shape().as_list()[0]
#        dim = Bottom.get_shape().as_list()[1] 
#        Bottom = tf.reshape(Bottom,[batchsize,dim,1,1])
#        
#        '''begin convolutional layer'''
#        conv1 = self._conv_layer(Bottom,128,128,(0.01,0.0),(1,2,1,1),'SAME','conv1')
#        bn1 = self._BatchNormalization(conv1,'bn1')
#        relu1  = self._ReLU(bn1)
#        max_pool1 = tf.nn.max_pool(relu1,(1,16,1,1),(1,2,1,1),'SAME')#256
#        
#        '''256 dim'''
#        res_1 = self._res_unit(max_pool1,32,256,True,'Res1')
#        res_2 = self._res_unit(res_1,32,256,False,'Res2')
#        max_pool2 = tf.nn.max_pool(res_2,(1,5,1,1),(1,2,1,1),'SAME')#128
#        '''128 dim'''
#        res_3 = self._res_unit(max_pool2,16,512,True,'Res3')
#        res_4 = self._res_unit(res_3,16,512,False,'Res4')
#        max_pool3 = tf.nn.max_pool(res_4,(1,5,1,1),(1,2,1,1),'SAME')#64
#        '''64 dim'''
#        res_5 = self._res_unit(max_pool3,8,512,True,'Res5')
#        res_6 = self._res_unit(res_5,8,512,False,'Res6')
#        max_pool4 = tf.nn.max_pool(res_6,(1,5,1,1),(1,2,1,1),'SAME')#32
#        '''32 dim'''
#        res_7 = self._res_unit(max_pool4,8,1024,True,'Res7')
#        res_8 = self._res_unit(res_7,8,1024,False,'Res8')
#        max_pool5 = tf.nn.max_pool(res_8,(1,5,1,1),(1,2,1,1),'SAME')#16
#        
#        resize = self._reshape_conv_to_fc(max_pool5)
#        fc1 = self._fully_connect_layer(resize,4096,(0.005,1.0),'ReLU','fc1')
#        fc2 = self._fully_connect_layer(fc1,4096,(0.005,1.0),'ReLU','fc2')
#
#        if self.Phase=='Train' and MMD=='Use':
#            
#            MMD1,Gramma1 = self._MMD_loss(fc2,0.3)
#        
#        fc3 = self._fully_connect_layer(fc2,10,(0.01,0),'None','output')
#        
#        if self.Phase=='Train' and MMD=='Use':
#            MMD2,Gamma2 = self._MMD_loss(fc3,0.3)                                          
#            return MMD1,MMD2,fc3
#        else:
#            return fc3                  

    def Loss(self,Bottom,Labels,Phase,MMD='UnUse'):                                           
        
        labels = tf.cast(Labels, tf.int64)
        
        if  MMD=='Use': 
            MMD1,MMD2,output = self.Build_CNN(Bottom,Phase,MMD)     
        else:
            output = self.Build_CNN(Bottom,Phase,MMD)                                   
        
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=output)                          
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)#######L1 L2 
#        reg_loss = tf.contrib.layers.apply_regularization(regularizer, reg_variables)#######L1 L2 

        tf.add_to_collection('losses', cross_entropy_mean)
#        tf.add_to_collection('losses', reg_loss)#######L1 L2 
        total_loss = tf.add_n(tf.get_collection('losses'), name='Label_Loss')  
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
#        opt = tf.train.GradientDescentOptimizer(1)
#        opt=tf.train.MomentumOptimizer(lr, momentum=0.9)                                            
        opt=tf.train.AdamOptimizer(lr)
        Grad = opt.compute_gradients(total_loss)  
#        capped_grads = [(tf.clip_by_norm(grad, clip_norm = 123.0, axes = 0)) for grad in Grad]           #dropout+max norm                             
        train_op = opt.apply_gradients(Grad,global_step=Global_step)
#        train_op = opt.apply_gradients(capped_grads)           #dropout+max norm 
        
        if MMD == 'Use':
            return train_op,total_loss,MMD1,MMD2  
        else:
            return train_op,total_loss                                       
    
    def Accuracy(self,Bottom,Labels):
        
        output = self.Build_CNN(Bottom,Phase='Test')
        
        ACC = tf.equal(tf.cast(tf.argmax(output,1),dtype=tf.uint8), 
                                      tf.cast(Labels,dtype=tf.uint8))
        ACC = tf.reduce_mean(tf.cast(ACC, "float")) 
        
        return output,ACC    
        
    def Detail_P6_ACC(self,Bottom,Labels):
        output = self.Build_CNN(Bottom,Phase='Test')
        
        ACC_Right = tf.equal(tf.cast(tf.argmax(output,1),dtype=tf.uint8), 
                                      tf.cast(Labels,dtype=tf.uint8))
        ACC_Right = tf.reduce_mean(tf.cast(ACC_Right, "float")) 
        
        ACC_Wrong_to_p2 = tf.equal(tf.cast(tf.argmax(output,1),dtype=tf.uint8), 
                                      tf.cast(tf.zeros_like(Labels),dtype=tf.uint8))
        ACC_Wrong_to_p2 = tf.reduce_mean(tf.cast(ACC_Wrong_to_p2, "float"))        

        ACC_Wrong_to_p3 = tf.equal(tf.cast(tf.argmax(output,1),dtype=tf.uint8), 
                                      tf.cast(tf.ones_like(Labels),dtype=tf.uint8))
        ACC_Wrong_to_p3 = tf.reduce_mean(tf.cast(ACC_Wrong_to_p3, "float"))   
        
        return ACC_Right,ACC_Wrong_to_p2,ACC_Wrong_to_p3,output