import tensorflow as tf

Alex_MEAN = [103.939, 116.779, 123.68]

class Res_ADDA():
    def __init__(self,pretrain,Initializer=None):
        self.pretrain = pretrain
        if self.pretrain:
            self.initializer = Initializer
        else:
            self.initializer = None
        return
    
    def _get_conv_filter(self,name,shape,dtype=tf.float32,initializer=None):#the shape(height,weight,in,output)###########################
               
#        if shape!=self.initializer[name]['weights'].shape:
#            print 'The params',shape,'and',self.initializer[name]['weights'].shape,'MisMatch!!'
#            exit()
#        else:
#            print 'Shape=',shape,'match:',self.initializer[name]['weights'].shape,'!!'
        if self.pretrain:
            weight = tf.get_variable(name=name+'_weights',initializer=self.initializer[name]['weights'])
        else:
            weight = tf.get_variable(name=name+'_weight',shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=dtype),
                            dtype=dtype)
            

        return weight################get_variable

    def _get_bias(self, name,shape,dtype=tf.float32,initializer=None,pretrain_close=False):
        
        
        if self.pretrain and pretrain_close==False:
            
            bias = tf.get_variable(name = name+'_biases',initializer=self.initializer[name]['biases'])
        else:
            bias = tf.get_variable(name=name+'_biases',shape=shape,
                        initializer=tf.constant_initializer(value=initializer, dtype=dtype),
                            dtype=dtype)

        return bias

    def _get_fc_weight(self, name,shape,dtype=tf.float32,initializer=None,pretrain_close=False):

        weight = tf.get_variable(name=name+'_weights',shape=shape,
                    initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=dtype),
                    dtype=dtype)
        
        return weight
        
    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID', name=name)    
        
        
    def _conv_layer(self,bottom,shape,name,pad,stride,dtype=tf.float32,initializer=(0.01,0),
                    group=1,wd=0.0005,bd=0.0):
#        with tf.variable_scope(name,reuse=None) as scope :
        weight = self._get_conv_filter(name,shape,dtype,initializer[0])
        conv_biases = self._get_bias(name,shape[-1],dtype,initializer[1])            
        if group == 1:
            conv = tf.nn.conv2d(bottom, weight, stride, padding=pad)
            bias = tf.nn.bias_add(conv, conv_biases)
        else:
            conv_groups = tf.split(bottom,group,3)
            weights_groups = tf.split(weight,group,3)
            conv = [tf.nn.conv2d(i,k, stride, padding=pad) for i, k in zip(conv_groups,weights_groups)]
            conv = tf.concat(conv,3)
            bias = tf.nn.bias_add(conv, conv_biases)
#            scope.reuse_variables()
            
        
        if self.Phase == 'Train':
            self._variable_decay(weight,wd)
            self._variable_decay(conv_biases,bd)
            

        return bias        
        
    def _RELU(self,bottom):
        relu = tf.nn.relu(bottom)
        return relu        
    
    
    def _LRN(self,bottom,name):
        norm = tf.nn.lrn(bottom,depth_radius=2,alpha=2e-05,beta=0.75,name=name)
        return norm        
    
    
    def _reshape_conv_to_fc(self,bottom):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
                 dim *= d
        x = tf.reshape(bottom, [-1, dim])
        return x
        
    def _fc_layer(self, bottom,output_num,name,
                 dtype=tf.float32,initializer=None,option=True,wd=0.0005,bd=0.0):
        
        dim = bottom.get_shape().as_list()[1]
        shape = (dim,output_num)
        
#        with tf.variable_scope(name,reuse=None) as scope:
            
        weights = self._get_fc_weight(name,shape,dtype,initializer[0],pretrain_close=option)
        biases = self._get_bias(name,output_num,dtype,initializer[1],pretrain_close=option)
        fc = tf.nn.bias_add(tf.matmul(bottom,weights),biases)
            
#            scope.reuse_variables()

        if self.Phase == 'Train':
            self._variable_decay(weights,wd)
            self._variable_decay(biases,bd)
        
        return fc
        


                       
    def _variable_decay(self,var,wd):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=var.op.name+'_loss')
        tf.add_to_collection('losses', weight_decay)
        return
    
    def _Dropout(self,bottom):
        if self.Phase=='Train':
            bottom = tf.nn.dropout(bottom,keep_prob=0.5)
        else:
            bottom = tf.nn.dropout(bottom,keep_prob=1.0)
        return bottom
#
    def _pre_process(self,DataBatch):
        
        bgr = DataBatch/255
#        # Convert RGB to BGR
#        red, green, blue = tf.split(3, 3, rgb_scaled)
#        assert rgb_scaled.get_shape().as_list()[1:] == [28, 28, 1]
#        assert green.get_shape().as_list()[1:] == [28, 28, 1]
#        assert blue.get_shape().as_list()[1:] == [28, 28, 1]#227
#        
#        bgr = tf.concat(3, [
#
#            red - Alex_MEAN[0],
#            green - Alex_MEAN[1],
#            blue - Alex_MEAN[2]
#            
#        ])
#        
#        
#        assert bgr.get_shape().as_list()[1:] == [28, 28, 3]
        return bgr        

    def Build_CNN_SOURCE(self,Bottom,Phase,reuse=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        self.Phase = Phase
        Bottom = self._pre_process(Bottom)

#        with tf.variable_scope('LeNet_Source'):
#            if reuse:
#                tf.get_variable_scope().reuse_variables()
        '''The First Scale Convolution Layers'''
        conv1 = self._conv_layer(bottom=Bottom, pad='VALID',
                             shape=(5,5,1,20),stride=[1,1,1,1],
                             initializer=(0.01,0),name="s_conv1",wd=0.0)
                                                            
        relu1 = self._RELU(conv1)
        pool1 = self._max_pool(relu1,name='s_pool1')
        norm1 = self._LRN(pool1,'s_norm1')
        
        ''' The Second Convolution Layer'''  
        conv2 = self._conv_layer(norm1,pad='VALID',
                                 shape=(5,5,20,50),stride=[1,1,1,1],
                                 initializer=(0.01,0),name="s_conv2",wd=0.0)
        relu2 = self._RELU(conv2)
        pool2 = self._max_pool(relu2,name='s_pool2')
        norm2 = self._LRN(pool2,'s_norm2')
        
        '''reshape'''
        reshape = self._reshape_conv_to_fc(norm2)
        
        '''Public FC layer1'''
        fc1 = self._fc_layer(reshape,
                           output_num=500, name='s_fc1',
                           dtype=tf.float32,initializer=(0.01,0.1))#0.005
                           
        fc1 = self._RELU(fc1)
        drop1 = self._Dropout(fc1)

        '''Public FC layer2'''
        fc2 = self._fc_layer(drop1,
                           output_num=500, name='s_fc2',
                           dtype=tf.float32,initializer=(0.01,0.1))#0.005
                           
        fc2 = self._RELU(fc2)  
        drop2 = self._Dropout(fc2)

        '''Public FC layer3'''
        fc3 = self._fc_layer(drop2,
                           output_num=128, name='s_fc3',
                           dtype=tf.float32,initializer=(0.01,0.1))#0.005
                           
#        fc3 = self._RELU(fc3)  
            
        return fc3

    def Build_CNN_TARGET(self,Bottom,Phase,reuse=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        self.Phase = Phase
     
        Bottom = self._pre_process(Bottom)
#        with tf.variable_scope('LeNet_Target'):
#            if reuse:
#                tf.get_variable_scope().reuse_variables()

        '''The First Scale Convolution Layers'''
        conv1 = self._conv_layer(bottom=Bottom, pad='VALID',
                                 shape=(5,5,1,20),stride=[1,1,1,1],
                                 initializer=(0.01,0),name="t_conv1",wd=0.0)
        relu1 = self._RELU(conv1)
        pool1 = self._max_pool(relu1,name='t_pool1')
        norm1 = self._LRN(pool1,'t_norm1')
           
        ''' The Second Convolution Layer'''  
        conv2 = self._conv_layer(norm1,pad='VALID',
                                 shape=(5,5,20,50),stride=[1,1,1,1],
                                 initializer=(0.01,0),name="t_conv2",wd=0.0)
        relu2 = self._RELU(conv2)
        pool2 = self._max_pool(relu2,name='t_pool2')
        norm2 = self._LRN(pool2,'t_norm2')

        reshape = self._reshape_conv_to_fc(norm2)
            
        '''Public FC layer1'''
        fc1 = self._fc_layer(reshape,
                           output_num=500, name='t_fc1',
                           dtype=tf.float32,initializer=(0.01,0.1))#0.005
        fc1 = self._RELU(fc1)  
        drop1 = self._Dropout(fc1)
         
        '''Public FC layer2'''
        fc2 = self._fc_layer(drop1,
                           output_num=500, name='t_fc2',
                           dtype=tf.float32,initializer=(0.01,0.1))#0.005
        fc2 = self._RELU(fc2)
        drop2 = self._Dropout(fc2)

        '''Public FC layer3'''
        fc3 = self._fc_layer(drop2,
                           output_num=128, name='t_fc3',
                           dtype=tf.float32,initializer=(0.01,0.1))#0.005
#        fc3 = self._RELU(fc3)
            
        return fc3
    
    def classifier(self,Bottom,reuse=None):
        
        if reuse:
           tf.get_variable_scope().reuse_variables()
            
        '''Classifier'''
        fc = self._fc_layer(Bottom,
                            output_num=10, name='c_classifier',
                            dtype=tf.float32,initializer=(0.01,0.1))       
        return fc  

    def discrimination(self,Bottom,reuse=None):

        with tf.variable_scope('dis',reuse=False) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
 
            ad1 = self._fc_layer(Bottom,
                       output_num=500, name='ad_1',
                       dtype=tf.float32,initializer=(0.01,0.1))
            ad1 = self._RELU(ad1) 
            ad2 = self._fc_layer(ad1,
                       output_num=500, name='ad_2',
                       dtype=tf.float32,initializer=(0.01,0.1))
            ad2 = self._RELU(ad2) 
            domain_output = self._fc_layer(ad2,
                           output_num=2, name='ad_3',
                           dtype=tf.float32,initializer=(0.01,0.1))
         
#            scope.reuse_variables()        
        
        return domain_output
#    def Adversarial_adaptation(self,source,target,Labels,base_lr,Global_step1,Global_step2,Global_step3):    

    def Loss(self,source,Labels_s,Bottom,Labels_t,base_lr,Global_step1,Global_step2):#,Global_step3):    
        
#        if Domain == 'source':
        '''pre_train the source data with label'''
            
        self.classifier_source= self.Build_CNN_SOURCE(source,'train')
        self.Label_output_source = self.classifier(self.classifier_source)
        self.labels_source = tf.cast(Labels_s, tf.int64)
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_source,logits=self.Label_output_source)                          
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        total_loss_source_label = tf.add_n(tf.get_collection('losses'), name='Label_Loss')
        Labels_ACC_pre_train = tf.equal(tf.cast(tf.argmax(self.Label_output_source,1),dtype=tf.uint8), 
                                      tf.cast(Labels_s,dtype=tf.uint8))
        Labels_ACC_pre_train = tf.reduce_mean(tf.cast(Labels_ACC_pre_train, "float"))  


               
        Label_output_target = self.Build_CNN_TARGET(Bottom,'train')                        
#        self.classifier_source= self.Build_CNN_SOURCE(source,'train',reuse=True)
#        reuse = None
#        domain_label_source = self.discrimination(self.classifier_source)## 111 source domain:0      !!!!!!!!!!!!!!!!!!    
#        domain_label_target = self.discrimination(Label_output_target,reuse=True)##target domain:1c???????????????????
        #Source:1,Target:0
        source_ft = self.classifier_source
        target_ft = Label_output_target

        classifier_ = tf.concat([source_ft,target_ft],0)
        source_adver_label = tf.zeros([tf.shape(source_ft)[0]],tf.int32)
        target_adver_label = tf.ones([tf.shape(target_ft)[0]],tf.int32)
        adver_label = tf.concat([source_adver_label,target_adver_label],0)#256
        adver_logits = self.discrimination(classifier_)## 111 source domain:0      !!!!!!!!!!!!!!!!!!    

        mapping_loss = tf.losses.sparse_softmax_cross_entropy(labels=1-adver_label,logits=adver_logits)
        adver_loss = tf.losses.sparse_softmax_cross_entropy(labels=adver_label,logits=adver_logits)
       
#        d_loss_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(domain_label_source),logits=domain_label_source))
#        d_loss_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(domain_label_target),logits=domain_label_target))       
#        map_target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(domain_label_target),logits=domain_label_target))
        
#        d_loss = d_loss_source + d_loss_target #+ map_target_loss
#        tf.add_to_collection('losses', d_loss)
#        total_loss_domain = tf.add_n(tf.get_collection('losses'), name='Domain_Loss')
            
#        map_target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(domain_label_target),logits=domain_label_target))
#        tf.add_to_collection('losses', map_target_loss)
#        map_loss = tf.add_n(tf.get_collection('losses'), name='target_Loss')
        
#        return total_loss_source_label,total_loss_domain,map_loss  
        lr1 = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step1,decay_steps=10000,decay_rate=0.5,staircase=True)
        lr2 = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step2,decay_steps=10000,decay_rate=0.5,staircase=True)
#        lr3 = tf.train.exponential_decay(learning_rate = base_lr,
#                                global_step=Global_step2,decay_steps=10000,decay_rate=0.5,staircase=True)
        t_vars = tf.trainable_variables()
        source_var = [var for var in t_vars if 's_' or 'c_' in var.name]
        target_var = [var for var in t_vars if 't_' in var.name]                                 
        discrim_var = [var for var in t_vars if 'ad_' in var.name]                                 
        with tf.variable_scope('optimize') :
            
#            opt1=tf.train.MomentumOptimizer(lr1, momentum=0.9) 
#            opt2=tf.train.MomentumOptimizer(lr2, momentum=0.9) 
#            opt3=tf.train.MomentumOptimizer(lr3, momentum=0.9)
            opt1=tf.train.AdamOptimizer(lr1) 
            opt2=tf.train.AdamOptimizer(lr2) 
#            opt3=tf.train.AdamOptimizer(lr3) 

#            opt1=tf.train.RMSPropOptimizer(lr1)#.minimize(total_loss_source_label,var_list=source_var)
#            opt2=tf.train.AdamOptimizer(lr2)#.minimize(map_loss,var_list=target_var)RMSPropOptimizer
#            D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
#            grad_d = tf.train.AdamOptimizer()#.minimize(d_loss,var_list=discrim_var)
#            grad_g = tf.train.AdamOptimizer()#.minimize(map_target_loss,var_list=target_var)

#            opt3=tf.train.RMSPropOptimizer(lr3)#.minimize(total_loss_domain,var_list=discrim_var)
#            D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
#                        .minimize(-D_loss, var_list=theta_D))
#            G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
#                        .minimize(G_loss, var_list=theta_G))
            source_Grad = opt1.compute_gradients(total_loss_source_label,var_list=source_var)
            Train_op0 = opt1.apply_gradients(source_Grad,global_step=Global_step1) ########################???????????????????????

            map_Grad = opt2.compute_gradients(mapping_loss,var_list=target_var)
            Train_op1 = opt2.apply_gradients(map_Grad) ########################???????????????????????
        
            discrim_Grad1 = opt2.compute_gradients(adver_loss,var_list=discrim_var) 
            Train_op2 = opt2.apply_gradients(discrim_Grad1) ########################???????????????????????
            
        return Train_op0,Labels_ACC_pre_train,Train_op2,Train_op1,total_loss_source_label,mapping_loss,adver_loss

            
            
            
     
    def Pre_training(self,Bottom,Labels,base_lr,Global_step):    
        
        total_loss = self.Loss(Bottom,Bottom,Labels,Domain='source')                               
        
        lr = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step,decay_steps=10000,decay_rate=0.5,staircase=True)
        
        t_vars = tf.trainable_variables()
        source_var = [var for var in t_vars if 's_' in var.name]

        opt=tf.train.AdamOptimizer(lr)
        Common_Grad = opt.compute_gradients(total_loss,var_list=source_var)
        Train_op = opt.apply_gradients(Common_Grad,global_step=Global_step)###############################  
        
        return Train_op,total_loss   

    def Adversarial_adaptation(self,source,target,Labels,base_lr,Global_step1,Global_step2,Global_step3):    
                
        total_loss_source_label,total_loss_domain,map_loss = self.Loss(source,target,Labels)#,Domain='target')                               
        
        lr1 = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step1,decay_steps=10000,decay_rate=0.5,staircase=True)
        lr2 = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step2,decay_steps=10000,decay_rate=0.5,staircase=True)
        lr3 = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step3,decay_steps=10000,decay_rate=0.5,staircase=True)
        
        t_vars = tf.trainable_variables()
        source_var = [var for var in t_vars if 's_' in var.name]
        target_var = [var for var in t_vars if 't_' in var.name]                                 
        discrim_var = [var for var in t_vars if 'ad_' in var.name]                                 
#        var = tf.concat([source_var,target_var,discrim_var],0)
#        loss = tf.concat([total_loss_source_label,map_loss,total_loss_domain],0)
#        with tf.variable_scope("Optimizer",reuse = None) as scope:
        with tf.variable_scope(tf.get_variable_scope(),reuse = None) :

            opt1=tf.train.AdamOptimizer(lr1).minimize(total_loss_source_label,var_list=source_var)
            opt2=tf.train.AdamOptimizer(lr2).minimize(map_loss,var_list=target_var)
            opt3=tf.train.AdamOptimizer(lr3).minimize(total_loss_domain,var_list=discrim_var)
        
#            source_Grad = opt1.compute_gradients(total_loss_source_label,var_list=source_var)
#            Train_op0 = opt1.apply_gradients(source_Grad,global_step=Global_step1) ########################???????????????????????

#            map_Grad = opt2.compute_gradients(map_loss,var_list=target_var)
#            Train_op1 = opt2.apply_gradients(map_Grad,global_step=Global_step2) ########################???????????????????????
        
#            discrim_Grad = opt3.compute_gradients(total_loss_domain,var_list=discrim_var) 
#            Train_op2 = opt3.apply_gradients(discrim_Grad,global_step=Global_step3) ########################???????????????????????

#        grad = opt.compute_gradients(discrim_Grad,var_list=discrim_var) 

#        grad = tf.concat([source_Grad,map_Grad,discrim_Grad],0)
#        Train_op2 = opt.apply_gradients(grad,global_step=Global_step) ########################???????????????????????
        
        return opt1,opt2,opt3,total_loss_source_label,total_loss_domain,map_loss
       
    def Accuracy(self,Bottom,Labels,Domain_Labels):
#          Dc_output,Lc_output_I  

        output = self.Build_CNN_TARGET(Bottom,Phase='Test')
        output = self.classifier(output,reuse=True)
                
        Labels_ACC = tf.equal(tf.cast(tf.argmax(output,1),dtype=tf.uint8), 
                                      tf.cast(Labels,dtype=tf.uint8))
        Labels_ACC = tf.reduce_mean(tf.cast(Labels_ACC, "float"))  
                          
        return Labels_ACC