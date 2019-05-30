import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
num=500

def creat_all_data():
    x=np.linspace(-10,10,num)

    noise=np.random.normal(0,8,x.shape)

    y=np.square(x)+5*x+noise

    x1=x.reshape(num,1)
    y1=y.reshape(num,1)
    return x1,y1

    
def load_quene_batch_data(batch_size,x_data,y_data):

    
    input_queue = tf.train.slice_input_producer([x_data,y_data], num_epochs=1,
                                                shuffle=True, capacity=batch_size)
##    num_threads=4和1运行时间一致？
    x_batch, y_batch = tf.train.batch(input_queue,
                                      batch_size=batch_size, num_threads=4, capacity=batch_size, allow_smaller_final_batch=False)

    return x_batch,y_batch

def load_batch_data(batch_size,x_data,y_data):

    batch_num=np.random.randint(0,num,batch_size)
    x_batch=np.zeros([batch_size,1])
    y_batch=np.zeros([batch_size,1])
    n=0
    for i in batch_num:
        x_batch[n]=x_data[i]
        y_batch[n]=y_data[i]
        n+=1

    return x_batch,y_batch


