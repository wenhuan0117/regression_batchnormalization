import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import creat_data
import time

def add_layer(inputs1,in_size,out_size,activation_function=None,batch_normalization=None,layer_name=None):

    if batch_normalization is None:
        inputs=inputs1
    else:
        inputs=batch_normalization(inputs1)
        
    with tf.variable_scope(layer_name) as scope:   
        weights=tf.Variable(name='weights',initial_value=tf.random_normal([in_size,out_size],stddev=1.0))
        biases=tf.Variable(name='biases',initial_value=tf.zeros([1,out_size])+0.1)
        
    Wx_plus_b=tf.matmul(inputs,weights)+biases

    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)

    return outputs

def batch_normalization(inputs):
    fc_mean,fc_var=tf.nn.moments(inputs,axes=[0],) ##axes=[0]根据需要修改
    outsize=inputs.get_shape()[1].value   ##[1]根据需要修改
    scale=tf.Variable(tf.ones([outsize]))
    shift=tf.Variable(tf.ones([outsize]))
    epsilon=0.001
    outputs=tf.nn.batch_normalization(inputs,fc_mean,fc_var,shift,scale,epsilon)
    return outputs

##define inputs
learning_rate_base=0.001
learning_rate_decay=0.99
learning_rate_decay_step=20
batch_size=100

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

x_data,y_data=creat_data.creat_all_data()


#def run with normalization

l1=add_layer(xs,1,30,activation_function=tf.nn.relu,batch_normalization=batch_normalization,layer_name='fc1')
l2=add_layer(l1,30,30,activation_function=tf.nn.relu,batch_normalization=batch_normalization,layer_name='fc2')
prediction=add_layer(l2,30,1,activation_function=None,layer_name='out')
global_step=tf.Variable(tf.constant(0))

learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,
                                        learning_rate_decay_step,learning_rate_decay)
cost=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

##是否分批训练0:no;1:yes
quene=0
start_time=time.time()
if quene==0:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.scatter(x_data,y_data)
        plt.ion()
        plt.xlim(-10,10)
        plt.ylim(-10,200)
        plt.show()
                
        for i in range(1000):
            #将训练数据放在此处，每次提取部分数据进行训练，类似dropout或者随机梯度下降法
            x_batch,y_batch=creat_data.load_batch_data(batch_size,x_data,y_data)
            sess.run(train_op,feed_dict={xs:x_batch,ys:y_batch})

            if i%20==0:
                pre_result,costresult=sess.run([prediction,cost],feed_dict={xs:x_data,ys:y_data})
                print(i,costresult)
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value=sess.run(prediction,feed_dict={xs:x_data})
                lines=ax.plot(x_data,prediction_value,'r-',lw=5)
                plt.pause(0.1)
else:
    
    with tf.Session() as sess:
##  必须放在tf.local_variables_initializer()之前  
        x_batch1,y_batch1=creat_data.load_quene_batch_data(batch_size,x_data,y_data)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())#very important!!!
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.scatter(x_data,y_data)
        plt.ion()
        plt.xlim(-10,10)
        plt.ylim(-10,200)
        plt.show()

        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess,coord)        
        epoch=0
        total_step=0
##对每个quene分开训练
        try:
            while not coord.should_stop():
                x_batch,y_batch=sess.run([x_batch1,y_batch1])
                for i in range(20):
                    sess.run(train_op,feed_dict={xs:x_batch,ys:y_batch})                
                    pre_result,costresult=sess.run([prediction,cost],feed_dict={xs:x_batch,ys:y_batch})
                    total_step+=1
                    print(epoch,i,total_step,costresult)
                    try:
                        ax.lines.remove(lines[0])
                    except Exception:
                        pass
                    prediction_value=sess.run(prediction,feed_dict={xs:x_data})
                    lines=ax.plot(x_data,prediction_value,'r-',lw=5)
                    plt.pause(0.1)
                epoch+=1
        except tf.errors.OutOfRangeError:
           print('done!')
        
        finally:    
            coord.request_stop()
        coord.join(threads)
end_time=time.time()
print("cost time:",end_time-start_time)



        
