from loader import load_data, seq_classes, seq_tokens
import json,random,os,time
import loader

import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

from data_helpers import *

from model import build_graph, num_epochs, batch_size


os.environ["CUDA_VISIBLE_DEVICES"]="3"

configs={
    # 'oh': {'embed': None, 'conv':None, 'rnn_depth':3},
    # 'embed_4': {'embed': 4, 'conv':None, 'rnn_depth':3},
    # 'embed_4_conv4': {'embed': 4, 'conv':4, 'rnn_depth':3}
    # 'embed_8_conv8': {'embed': 8, 'conv':8, 'rnn_depth':2}
    # 'embed_4_depth1': {'embed': 4, 'conv':None, 'rnn_depth':1},
    # 'embed_4_depth1_batch8': {'embed': 4, 'conv':None, 'rnn_depth':1},
    'embed_4_depth3_batch32_epoch20': {'embed': 4, 'conv':None, 'rnn_depth':2},
    # 'embed_4_depth3_batch8': {'embed': 4, 'conv':None, 'rnn_depth':3},
    # 'oh_depth1': {'embed': None, 'conv':None, 'rnn_depth':1},
    # 'oh_depth2': {'embed': None, 'conv':None, 'rnn_depth':2}
}
num_attempts = 3
to_restore=False
results={name:{} for name in configs.keys()}
start=time.time()
for model_id, cfg in configs.items():
    for attempt in range(0,num_attempts):
        print('Training: '+ model_id + ' #'+str(attempt))
        tf.reset_default_graph()

        x,y,opt, accuracy, y_hat, loss, embedding_encoder,_,dropout_active = build_graph(cfg['embed'], cfg['conv'], cfg['rnn_depth'])

        chkpt_path='./models/'+ model_id +'-'+str(attempt)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
            saver = tf.train.Saver()
            if to_restore:
                saver.restore(sess, chkpt_path+ '/model.checkpoint')
            else:
                sess.run(tf.global_variables_initializer())

            best_dev_acc=0

            for e in range(num_epochs):
                print('Epoch ', e, time.time()-start)
                start=time.time()
                i=0
                # random.shuffle(train_data)
                while (i+1)*batch_size < len(train_data):
                    # run an optimisation step
                    # batch_xs, batch_ys = get_batch(train_data, i*batch_size, batch_size)
                    batch_xs, batch_ys = get_batch_reweighted(train_data_dict, batch_size, training_class_ratios)

                    _, this_loss, this_acc,pred = sess.run([opt, loss, accuracy,y_hat], feed_dict={x:batch_xs, y:batch_ys, dropout_active:True})

                    i += 1
                    if i % 50 ==0:
                        dev_accs =[]
                        for j in range(len(dev_data)//batch_size):
                            batch_xs, batch_ys = get_batch(dev_data, j*batch_size, batch_size)
                            dev_acc,pred= sess.run([accuracy,y_hat], feed_dict={x:batch_xs, y:batch_ys})
                            dev_accs.append(dev_acc)
                        # print(e,i,this_loss,this_acc,np.mean(dev_accs) )
                        # print(pred, batch_ys)
                        # print(batch_xs)
                        if np.mean(dev_accs) > best_dev_acc:
                            print('New best of {}! Saving ({})'.format(np.mean(dev_accs), np.mean(this_acc)))
                            if not os.path.exists(chkpt_path):
                                os.makedirs(chkpt_path)
                            saver.save(sess, chkpt_path+'/model.checkpoint')
                            best_dev_acc = np.mean(dev_accs)
                            results[model_id][attempt] = best_dev_acc
print(results)
with open('./models/results.json', 'w') as fp:
    json.dump(results, fp)
