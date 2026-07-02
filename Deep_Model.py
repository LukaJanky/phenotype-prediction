import warnings
import os
import time
import signal
import sys
import copy
import h5py
import pickle
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Setari experiment
FLAGS = tf.app.flags.FLAGS 

# -- Configurarea mediului --
tf.app.flags.DEFINE_string('log_dir', "F:\licenta\Licenta\Python", "log_dir")
tf.app.flags.DEFINE_string('dir_data', "", "folderul care contine fisierele pentru antrenament si evaluare")
tf.app.flags.DEFINE_bool('save', False, "salvare model")
tf.app.flags.DEFINE_bool('restore', False, "Restituire model anterior")
tf.app.flags.DEFINE_bool('is_training', True, "model antrenabil")
tf.app.flags.DEFINE_string('processing', "train", "actiuni aplicabile modelului {train,evaluate,predict}")

# -- Arhitectura retelei neuronale --
tf.app.flags.DEFINE_integer('n_input', 54675, "numar de trasaturi")
tf.app.flags.DEFINE_integer('n_classes', 1, "numar de clase")
tf.app.flags.DEFINE_integer('n_layers', 6, "numar de straturi")
tf.app.flags.DEFINE_integer('n_hidden_1', 1574, "numarul de neuroni din primul strat ascuns") #Nivelul 7
tf.app.flags.DEFINE_integer('n_hidden_2', 1386, "numarul de neuroni din al doilea strat ascuns") #Nivelul 6
tf.app.flags.DEFINE_integer('n_hidden_3', 951, "numarul de neuroni din al treilea strat ascuns") #Nivelul 5
tf.app.flags.DEFINE_integer('n_hidden_4', 515, "numarul de neuroni din al patrulea strat ascuns") #Nivelul 4
tf.app.flags.DEFINE_integer('n_hidden_5', 255, "numarul de neuroni din al cincilea strat ascuns") #Nivelul 3
tf.app.flags.DEFINE_integer('n_hidden_6', 90, "numarul de neuroni din al saselea strat ascuns") #Nivelul 2

# -- Invatare si hiperparamentrii --
tf.app.flags.DEFINE_string('lr_method', 'adam', "optimizer {adam, momentum, adagrad, rmsprop}")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "rata initiala de invatare")
tf.app.flags.DEFINE_bool('bn', False, "utilizare normalizare batch-uri")
tf.app.flags.DEFINE_float('keep_prob', 0.4, "probabilitatea de pastrare pentru dropout")
tf.app.flags.DEFINE_string('type_training', 'LGO', "termen de regularizare {"", LGO, L2, L1}")
tf.app.flags.DEFINE_float('alpha', 1, "valoarea hiperparametrului alfa")
tf.app.flags.DEFINE_integer('display_step', 5, "cand sa se printeze performantele")
tf.app.flags.DEFINE_integer('batch_size', 2**6, "numarul de exemple dintr-un batch")
tf.app.flags.DEFINE_integer('epochs', 20, "numarul de epoci pentru antrenamant")
tf.app.flags.DEFINE_string('GPU_device', '/gpu:0', "GPU device")

from base_model import BaseModel

def l1_loss_func(x):
    return tf.reduce_sum(tf.math.abs(x)) 

def l2_loss_func(x):
    return tf.reduce_sum(tf.square(x))

def train(save_dir):

    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_device[len(FLAGS.GPU_device)-1]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # Incarcare matrici de corespondente pentru ontologia genei
    print("Loading the connection matrix...")
    start = time.time()

    adj_matrix = pd.read_csv(os.path.abspath(os.path.join(FLAGS.dir_data,"adj_matrix.csv")),index_col=0)
    first_matrix_connection = pd.read_csv(os.path.abspath(os.path.join(FLAGS.dir_data,"first_matrix_connection_GO.csv")),index_col=0)
    csv_go = pd.read_csv(os.path.abspath(os.path.join(FLAGS.dir_data,"go_level.csv")),index_col=0)

    connection_matrix = []
    connection_matrix.append(np.array(first_matrix_connection.values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(7)].loc[lambda x: x==1].index,csv_go[str(6)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(6)].loc[lambda x: x==1].index,csv_go[str(5)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(5)].loc[lambda x: x==1].index,csv_go[str(4)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(4)].loc[lambda x: x==1].index,csv_go[str(3)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(3)].loc[lambda x: x==1].index,csv_go[str(2)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.ones((FLAGS.n_hidden_6, FLAGS.n_classes),dtype=np.float32))
    end = time.time()
    elapsed=end - start
    print("Total time: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
    time.gmtime(elapsed).tm_min,
    time.gmtime(elapsed).tm_sec))

    # Incarcare date
    print("Loading the data...")

    start = time.time()
    loaded = np.load(os.path.abspath(os.path.join(FLAGS.dir_data,"X_train.npz")))
    X_train = loaded['x']
    X_train = np.transpose(X_train)
    y_train = loaded['y']
    y_train = y_train.reshape(-1,1)
    if FLAGS.n_classes>=2:
        y_train=to_categorical(y_train)

    loaded = np.load(os.path.abspath(os.path.join(FLAGS.dir_data,"X_test.npz")))
    X_test = loaded['x']
    X_test = np.transpose(X_test)
    y_test = loaded['y']
    y_test = y_test.reshape(-1,1)
    if FLAGS.n_classes>=2:
        y_test=to_categorical(y_test)

    end = time.time()
    elapsed=end - start
    print("Total time: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
    time.gmtime(elapsed).tm_min,
    time.gmtime(elapsed).tm_sec))


    # Lansare model
    print("Launching the learning")
    if FLAGS.type_training != "":
        print("with {} and ALPHA={}".format(FLAGS.type_training,FLAGS.alpha))

    tf.reset_default_graph() 
   
    # -- Intrarile modelului --
    X = tf.placeholder(tf.float32, shape=[None, FLAGS.n_input])
    Y = tf.placeholder(tf.float32, shape=[None, FLAGS.n_classes])
    # -- Hiperparametrii retelei neuronale --
    is_training = tf.placeholder(tf.bool,name="is_training") # Batch Norm hyperparameter
    learning_rate = tf.placeholder(tf.float32, name="learning_rate") # Optimizer hyperparameter
    keep_prob = tf.placeholder(tf.float32, name="keep_prob") # Dropout hyperparameter
    total_batches=len(X_train)//FLAGS.batch_size

    network=BaseModel(X=X,n_input=FLAGS.n_input,n_classes=FLAGS.n_classes,
        n_hidden_1=FLAGS.n_hidden_1,n_hidden_2=FLAGS.n_hidden_2,n_hidden_3=FLAGS.n_hidden_3,n_hidden_4=FLAGS.n_hidden_4,
        n_hidden_5=FLAGS.n_hidden_5,n_hidden_6=FLAGS.n_hidden_6,keep_prob=keep_prob,is_training=is_training) # Instantiere model
    pred = network()

    # -- Functia cost --

    # ---- Cross-Entropy loss  ----
    # calcularea functiei cost pentru toate dimensiunile
    if FLAGS.n_classes>=2:
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y)) 
    else:
        ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y))
    
    # ---- Regularizare (LGO, L2, L1) ----
    additional_loss = 0
    if FLAGS.type_training=="LGO":
        for idx,weight in enumerate(network.weights.values()):
            additional_loss+=l2_loss_func(weight*(1-connection_matrix[idx])) #Penalizarea legaturilor noGO
    elif FLAGS.type_training=="L2" :
        for weight in network.weights.values():
            additional_loss += l2_loss_func(weight)
    elif FLAGS.type_training=="L1" :
        for idx,weight in enumerate(network.weights.values()):
            additional_loss+=l1_loss_func(weight)
            
    # ---- Total cost  ----
    if FLAGS.type_training!="" :
        total_loss = ce_loss + FLAGS.alpha*additional_loss
    else:
        total_loss = ce_loss
    
    # ---- Norma pentru ponderile conexiunilor ----
    norm_no_go_connections=0
    norm_go_connections=0
    for idx,weight in enumerate(list(network.weights.values())[:-1]):
        norm_no_go_connections+=tf.norm((weight*(1-connection_matrix[idx])),ord=1)/np.count_nonzero(1-connection_matrix[idx])
        norm_go_connections+=tf.norm((weight*connection_matrix[idx]),ord=1)/np.count_nonzero(connection_matrix[idx])
    norm_no_go_connections/=FLAGS.n_layers
    norm_go_connections/=FLAGS.n_layers

    # -- Tehnica de optimizare --
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        if FLAGS.lr_method=="adam":
            trainer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        elif FLAGS.lr_method=="momentum":
            trainer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=0.09, use_nesterov=True)
        elif FLAGS.lr_method=="adagrad":
            trainer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif FLAGS.lr_method=="rmsprop":
            trainer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
        optimizer = trainer.minimize(total_loss)

    # -- Calcularea erorii de predictie --
    if FLAGS.n_classes>=2:
        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y, 1))
    else:
        sig_pred=tf.nn.sigmoid(pred)
        sig_pred=tf.cast(sig_pred>0.5,dtype=tf.int64)
        ground_truth=tf.cast(Y,dtype=tf.int64)
        correct_prediction = tf.equal(sig_pred,ground_truth)
    
    # -- Calcularea acuratetei pentru toate batch-urile si media acestora --
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

    # -- Initializare variabile --
    init = tf.global_variables_initializer()

    # -- Configurare utilizare GPU --
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    #config.gpu_options.allow_growth = True, log_device_placement=True

    if FLAGS.save or FLAGS.restore : saver = tf.train.Saver()

    start = time.time()

    with tf.device(FLAGS.GPU_device):
        with tf.Session(config=config) as sess: 
            sess.run(init)

            train_c_accuracy=[]
            train_c_total_loss=[]

            test_c_accuracy=[]
            test_c_total_loss=[]

            c_l1_norm_go=[]
            c_l1_norm_no_go=[]

            if FLAGS.type_training!="":
                train_c_ce_loss=[]
                test_c_ce_loss=[]
                train_c_additional_loss=[]
                test_c_additional_loss=[]

            for epoch in tqdm(np.arange(0,FLAGS.epochs)):

                index = np.arange(X_train.shape[0])
                np.random.shuffle(index)
                batch_X = np.array_split(X_train[index], total_batches)
                batch_Y = np.array_split(y_train[index], total_batches)

                # -- Optimizare --
                for batch in range(total_batches):
                    batch_x,batch_y=batch_X[batch],batch_Y[batch]
                    sess.run(optimizer, feed_dict={X: batch_x,Y: batch_y,is_training:FLAGS.is_training,keep_prob:FLAGS.keep_prob,learning_rate:FLAGS.learning_rate})

                if ((epoch+1) % FLAGS.display_step == 0) or (epoch==0) :
                    if not((FLAGS.display_step==FLAGS.epochs) and (epoch==0)):

                        # -- Calcularea acuratetei si costului pentru batch-uri dupa o epoca pe seturile de train si test --

                        avg_cost,avg_acc,l1_norm_no_go,l1_norm_go = sess.run([total_loss, accuracy,norm_no_go_connections,norm_go_connections], feed_dict={X: X_train,Y: y_train,
                                                               is_training:False,keep_prob:1.0})
                        train_c_total_loss.append(avg_cost)
                        train_c_accuracy.append(avg_acc)
                        c_l1_norm_go.append(l1_norm_go)
                        c_l1_norm_no_go.append(l1_norm_no_go)

                        if FLAGS.type_training!="":
                            avg_ce_loss,avg_additional_loss= sess.run([ce_loss, additional_loss], feed_dict={X: X_train,Y: y_train,is_training:False,keep_prob:1.0})
                            train_c_additional_loss.append(avg_additional_loss)
                            train_c_ce_loss.append(avg_ce_loss)

                        avg_cost,avg_acc = sess.run([total_loss, accuracy], feed_dict={X: X_test,Y: y_test,is_training:False,keep_prob:1.0})
                        test_c_total_loss.append(avg_cost)
                        test_c_accuracy.append(avg_acc)

                        if FLAGS.type_training!="": 
                            avg_ce_loss,avg_additional_loss= sess.run([ce_loss, additional_loss], feed_dict={X: X_test,Y: y_test,is_training:False,keep_prob:1.0})
                            test_c_additional_loss.append(avg_additional_loss)
                            test_c_ce_loss.append(avg_ce_loss)                

                        current_idx=len(train_c_total_loss)-1                   
                        print('| Epoch: {}/{} | Train: Loss {:.6f} Accuracy : {:.6f} '\
                        '| Test: Loss {:.6f} Accuracy : {:.6f}\n'.format(
                        epoch+1, FLAGS.epochs,train_c_total_loss[current_idx], train_c_accuracy[current_idx],test_c_total_loss[current_idx],test_c_accuracy[current_idx]))

            if FLAGS.save: saver.save(sess=sess, save_path=os.path.join(save_dir,"model"))

    end = time.time()
    elapsed=end - start
    print("Total time: {}h {}min {}sec ".format(time.gmtime(elapsed).tm_hour,
    time.gmtime(elapsed).tm_min,
    time.gmtime(elapsed).tm_sec))   

    performances = {
                    'total_loss':train_c_total_loss,'test_total_loss':test_c_total_loss,
                    'acc':train_c_accuracy,'test_acc':test_c_accuracy
                    }

    performances['norm_go']=c_l1_norm_go
    performances['norm_no_go']=c_l1_norm_no_go

    if FLAGS.type_training!="":      
        performances['additional_loss']=train_c_additional_loss
        performances['test_additional_loss']=test_c_additional_loss
        performances['ce_loss']=train_c_ce_loss
        performances['test_ce_loss']=test_c_ce_loss


    return performances

def evaluate(save_dir):

    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_device[len(FLAGS.GPU_device)-1]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # Incarcare matrici de corespondente pentru ontologia genei
    print("Loading the connection matrix...")
    start = time.time()

    adj_matrix = pd.read_csv(os.path.join(FLAGS.dir_data,"adj_matrix.csv"),index_col=0)
    first_matrix_connection = pd.read_csv(os.path.join(FLAGS.dir_data,"first_matrix_connection_GO.csv"),index_col=0)
    csv_go = pd.read_csv(os.path.join(FLAGS.dir_data,"go_level.csv"),index_col=0)

    connection_matrix = []
    connection_matrix.append(np.array(first_matrix_connection.values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(7)].loc[lambda x: x==1].index,csv_go[str(6)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(6)].loc[lambda x: x==1].index,csv_go[str(5)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(5)].loc[lambda x: x==1].index,csv_go[str(4)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(4)].loc[lambda x: x==1].index,csv_go[str(3)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(3)].loc[lambda x: x==1].index,csv_go[str(2)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.ones((FLAGS.n_hidden_6, FLAGS.n_classes),dtype=np.float32))

    end = time.time()
    elapsed=end - start
    print("Total time: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
    time.gmtime(elapsed).tm_min,
    time.gmtime(elapsed).tm_sec))

    # Incarcare date
    print("Loading the test dataset...")

    loaded = np.load(os.path.join(FLAGS.dir_data,"X_prediction.npz"))
    X_test = loaded['x']
    X_test = np.transpose(X_test)
    y_test = loaded['y']
    y_test = y_test.reshape(-1,1)
    if FLAGS.n_classes>=2:
        y_test=to_categorical(y_test)

    end = time.time()
    elapsed=end - start
    print("Total time: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
    time.gmtime(elapsed).tm_min,
    time.gmtime(elapsed).tm_sec))

    # Lansare model
    print("Launching the evaluation")
    if FLAGS.type_training != "":
        print("with {} and ALPHA={}".format(FLAGS.type_training,FLAGS.alpha))

    tf.reset_default_graph() 
   
    # -- Intrarile modelului --
    X = tf.placeholder(tf.float32, shape=[None, FLAGS.n_input])
    Y = tf.placeholder(tf.float32, shape=[None, FLAGS.n_classes])

    # -- Hiperparametrii retelei neuronale --
    is_training = tf.placeholder(tf.bool,name="is_training") # Batch Norm hyperparameter
    keep_prob = tf.placeholder(tf.float32, name="keep_prob") # Dropout hyperparameter

    network=BaseModel(X=X,n_input=FLAGS.n_input,n_classes=FLAGS.n_classes,
        n_hidden_1=FLAGS.n_hidden_1,n_hidden_2=FLAGS.n_hidden_2,n_hidden_3=FLAGS.n_hidden_3,n_hidden_4=FLAGS.n_hidden_4,
        n_hidden_5=FLAGS.n_hidden_5,n_hidden_6=FLAGS.n_hidden_6,keep_prob=keep_prob,is_training=is_training) # Model instantiation
    pred = network()

    # -- Functia cost --

    # ---- Cross-Entropy loss  ----
    # calcularea functiei cost pentru toate dimensiunile
    if FLAGS.n_classes>=2:
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y)) 
    else:
        ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y))
    
    # ---- Regularizare (LGO, L2, L1) ----
    additional_loss = 0
    if FLAGS.type_training=="LGO":
        for idx,weight in enumerate(network.weights.values()):
            additional_loss+=l2_loss_func(weight*(1-connection_matrix[idx])) #Penalizarea legaturilor noGO
    elif FLAGS.type_training=="L2" :
        for weight in network.weights.values():
            additional_loss += l2_loss_func(weight)
    elif FLAGS.type_training=="L1" :
        for idx,weight in enumerate(network.weights.values()):
            additional_loss+=l1_loss_func(weight)

    # ---- Total loss  ----
    if FLAGS.type_training!='' :
        total_loss = ce_loss + FLAGS.alpha*additional_loss
    else:
        total_loss = ce_loss

    # -- Calcularea erorii de predictie --
    if FLAGS.n_classes>=2:
        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y, 1))
    else:
        sig_pred=tf.nn.sigmoid(pred)
        sig_pred=tf.cast(sig_pred>0.5,dtype=tf.int64)
        ground_truth=tf.cast(Y,dtype=tf.int64)
        correct_prediction = tf.equal(sig_pred,ground_truth)

    # -- Calcularea acuratetei pentru toate batch-urile si media acestora --
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

    # -- Configurare utilizare GPU --
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    #config.gpu_options.allow_growth = True, log_device_placement=True

    if FLAGS.restore : saver = tf.train.Saver()

    start = time.time()

    with tf.device(FLAGS.GPU_device):
        with tf.Session(config=config) as sess: 
            if  FLAGS.restore:
                saver.restore(sess,os.path.join(save_dir,"model")) 

            # -- Calcularea acuratetei si costului pentru datele de test --

            avg_cost,avg_acc = sess.run([total_loss, accuracy], feed_dict={X: X_test,Y: y_test,is_training:False,keep_prob:1.0})          
         
            print('Test loss {:.6f}, test accuracy : {:.6f}\n'.format(avg_cost,avg_acc))

    end = time.time()
    elapsed=end - start
    print("Total time: {}h {}min {}sec ".format(time.gmtime(elapsed).tm_hour,
    time.gmtime(elapsed).tm_min,
    time.gmtime(elapsed).tm_sec))   

    return

def predict(save_dir):

    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_device[len(FLAGS.GPU_device)-1]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # Incarcare matrici de corespondente pentru ontologia genei
    print("Loading the connection matrix...")
    start = time.time()

    adj_matrix = pd.read_csv(os.path.join(FLAGS.dir_data,"adj_matrix.csv"),index_col=0)
    first_matrix_connection = pd.read_csv(os.path.join(FLAGS.dir_data,"first_matrix_connection_GO.csv"),index_col=0)
    csv_go = pd.read_csv(os.path.join(FLAGS.dir_data,"go_level.csv"),index_col=0)

    connection_matrix = []
    connection_matrix.append(np.array(first_matrix_connection.values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(7)].loc[lambda x: x==1].index,csv_go[str(6)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(6)].loc[lambda x: x==1].index,csv_go[str(5)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(5)].loc[lambda x: x==1].index,csv_go[str(4)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(4)].loc[lambda x: x==1].index,csv_go[str(3)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(3)].loc[lambda x: x==1].index,csv_go[str(2)].loc[lambda x: x==1].index].values,dtype=np.float32))
    connection_matrix.append(np.ones((FLAGS.n_hidden_6, FLAGS.n_classes),dtype=np.float32))

    end = time.time()
    elapsed=end - start
    print("Total time: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
    time.gmtime(elapsed).tm_min,
    time.gmtime(elapsed).tm_sec))

    # Incarcare date
    print("Loading the test dataset...")

    loaded = np.load(os.path.join(FLAGS.dir_data,"X_test.npz"))
    X_test = loaded['x']
    X_test = np.transpose(X_test)
    y_test = loaded['y']
    y_test = y_test.reshape(-1,1)
    if FLAGS.n_classes>=2:
        y_test=to_categorical(y_test)

    end = time.time()
    elapsed=end - start
    print("Total time: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
    time.gmtime(elapsed).tm_min,
    time.gmtime(elapsed).tm_sec))

    # Lansare model
    print("Launching the evaluation")
    if FLAGS.type_training != "":
        print("with {} and ALPHA={}".format(FLAGS.type_training,FLAGS.alpha))

    tf.reset_default_graph() 
   
    # -- Intrarile modelului --
    X = tf.placeholder(tf.float32, shape=[None, FLAGS.n_input])
    Y = tf.placeholder(tf.float32, shape=[None, FLAGS.n_classes])

    # -- Hiperparametrii retelei neuronale --
    is_training = tf.placeholder(tf.bool,name="is_training") # Batch Norm hyperparameter
    keep_prob = tf.placeholder(tf.float32, name="keep_prob") # Dropout hyperparameter

    network=BaseModel(X=X,n_input=FLAGS.n_input,n_classes=FLAGS.n_classes,
        n_hidden_1=FLAGS.n_hidden_1,n_hidden_2=FLAGS.n_hidden_2,n_hidden_3=FLAGS.n_hidden_3,n_hidden_4=FLAGS.n_hidden_4,
        n_hidden_5=FLAGS.n_hidden_5,n_hidden_6=FLAGS.n_hidden_6,keep_prob=keep_prob,is_training=is_training) # Instantiere model
    pred = network()
    # -- Calcularea erorii de predictie --
    if FLAGS.n_classes>=2:
        y_hat = tf.argmax(pred,1)
    else:
        y_hat = tf.nn.sigmoid(pred)
        y_hat = tf.cast(pred>0.5,dtype=tf.int64)

    # -- Configurare utilizare GPU --
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    #config.gpu_options.allow_growth = True, log_device_placement=True

    if FLAGS.restore : saver = tf.train.Saver()

    start = time.time()

    with tf.device(FLAGS.GPU_device):
        with tf.Session(config=config) as sess: 
            if  FLAGS.restore:
                saver.restore(sess,os.path.join(save_dir,"model")) 
            
            # -- Efectuarea predictiilor pentru esantioanele din setul pentru testare --

            y_hat = sess.run([y_hat], feed_dict={X: X_test,Y: y_test,is_training:False,keep_prob:1.0})          

    end = time.time()
    elapsed=end - start
    print("Total time: {}h {}min {}sec ".format(time.gmtime(elapsed).tm_hour,
    time.gmtime(elapsed).tm_min,
    time.gmtime(elapsed).tm_sec))   

    return y_hat

def main(_):

    save_dir=os.path.join(FLAGS.log_dir,'MLP_DP={}_BN={}_EPOCHS={}_OPT={}'.format(FLAGS.keep_prob,FLAGS.bn,FLAGS.epochs,FLAGS.lr_method))

    if FLAGS.type_training!="" :
        save_dir=save_dir+'_{}_ALPHA={}'.format(FLAGS.type_training,FLAGS.alpha)
    
    if FLAGS.processing=="train":

        start_full = time.time()
        
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        performances=train(save_dir=save_dir)   

        with open(os.path.join(save_dir,"histories.txt"), "wb") as fp:
            #Pickling
            pickle.dump(performances, fp)

        end = time.time()
        elapsed =end - start_full
        print("Total time full process: {}h {}min {}sec".format(time.gmtime(elapsed).tm_hour,
        time.gmtime(elapsed).tm_min,
        time.gmtime(elapsed).tm_sec))
        
    elif FLAGS.processing=="evaluate":
        
        evaluate(save_dir=save_dir)
        
    elif FLAGS.processing=="predict":
        
        np.savez_compressed(os.path.join(save_dir,'y_test_hat'),y_hat=predict(save_dir=save_dir))
    
if __name__ == "__main__":
    tf.app.run()