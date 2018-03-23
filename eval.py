from loader import load_data, seq_classes, seq_tokens
import json,random,os
import loader

import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

from data_helpers import *
from model import build_graph, batch_size



os.environ["CUDA_VISIBLE_DEVICES"]="2"

configs={
    # 'oh': {'embed': None, 'conv':None, 'rnn_depth':3},
    # 'embed_4': {'embed': 4, 'conv':None, 'rnn_depth':3},
    # 'embed_4_conv4': {'embed': 4, 'conv':4, 'rnn_depth':3}
    # 'embed_8_conv8': {'embed': 8, 'conv':8, 'rnn_depth':2}
    # 'embed_4_conv4_depth1': {'embed': 4, 'conv':4, 'rnn_depth':1},
    # 'embed_4_conv4_depth2': {'embed': 4, 'conv':4, 'rnn_depth':2},
    # 'oh_depth1': {'embed': None, 'conv':None, 'rnn_depth':1},
    # 'oh_depth2': {'embed': None, 'conv':None, 'rnn_depth':2}

    'embed_4_depth2': {'embed': 4, 'conv':None, 'rnn_depth':2}
}
num_attempts = 3
to_restore=False
results={name:{} for name in configs.keys()}



if False:
    for model_id, cfg in configs.items():
        for attempt in range(0,num_attempts):
            tf.reset_default_graph()
            x,y,opt, accuracy, y_hat, loss, embedding_encoder,_,dropout_active = build_graph(cfg['embed'], cfg['conv'], cfg['rnn_depth'])
            chkpt_path='./models/'+ model_id +'-'+str(attempt)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
                saver = tf.train.Saver()

                saver.restore(sess, chkpt_path+'/model.checkpoint')
                best_dev_acc=0
                print('Final model evaluation: ', model_id, attempt)

                # train_accs =[]
                # for j in range(len(train_data[:2000])//batch_size):
                #     batch_xs, batch_ys = get_batch(train_data, j, batch_size)
                #     train_acc,pred= sess.run([accuracy,y_hat], feed_dict={x:batch_xs, y:batch_ys})
                #     train_accs.append(train_acc)
                # print('Train: ', np.mean(train_accs) )

                dev_accs =[]
                dev_preds=[]
                dev_gold=[]
                for j in range(len(dev_data)//batch_size):
                    batch_xs, batch_ys = get_batch(dev_data, j*batch_size, batch_size)
                    dev_acc,dev_pred= sess.run([accuracy,y_hat], feed_dict={x:batch_xs, y:batch_ys})
                    dev_accs.append(dev_acc)
                    dev_gold.extend(batch_ys)
                    # print(dev_pred.tolist())
                    # exit()
                    dev_preds.extend(dev_pred.tolist())
                print('Dev: ', np.mean(dev_accs) )

# test_data = dev_data

model_id='embed_4_depth2'
attempt=1
cfg = configs[model_id]
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
tf.reset_default_graph()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
    chkpt_path='./models/'+ model_id +'-'+str(attempt)
    x,y,opt, accuracy, y_hat, loss, embedding_encoder,rnn_out,dropout_active = build_graph(cfg['embed'], cfg['conv'], cfg['rnn_depth'])

    saver = tf.train.Saver()

    saver.restore(sess, chkpt_path+ '/model.checkpoint')
    best_dev_acc=0
    print('Final model evaluation:')


    # Only check the test set after model selection!
    test_accs =[]
    test_preds=[]
    test_gold = []
    for j in range(len(test_data)//batch_size):
        batch_xs, batch_ys = get_batch(test_data, j*batch_size, batch_size)
        test_acc,test_pred= sess.run([accuracy,y_hat], feed_dict={x:batch_xs, y:batch_ys})
        test_accs.append(test_acc)
        test_preds.extend(test_pred.tolist())
        test_gold.extend(batch_ys)
    print('Test: ' ,np.mean(test_accs) )



    test_preds=np.asarray(test_preds)

    # eval blind data

    classes_pretty={0:'Cytosolic', 1:'Mitochondrial', 2:'Nuclear', 3:'Secreted'}

    batch_xs,_ = get_batch(blind_data, 0, len(blind_data))
    test_pred= sess.run(y_hat, feed_dict={x:batch_xs})
    pred_dict = {blind_data[i][1]:test_pred[i] for i in range(len(blind_data))}
    print('Seq. Id & Predicted class & Prediction probability \\\\ \n\hline')
    for seq,probs in pred_dict.items():
        print(seq, ' & ',classes_pretty[np.argmax(probs)], ' & ', "{:0.3f}".format(np.amax(probs)) , ' \\\\')


    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(np.equal(test_gold,i, dtype=int), test_preds[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])


    plt.figure()
    lw = 2
    for c in range(num_classes):
        plt.plot(fpr[c], tpr[c],
             lw=lw, label='{:} (area = {:0.2f})'.format( seq_classes_rev[c], roc_auc[c]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()

    plt.figure()
    lw = 2
    for c in range(num_classes):
        plt.plot(fpr[c], tpr[c],
             lw=lw, label='{:} (area = {:0.2f})'.format( seq_classes_rev[c], roc_auc[c]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 0.05])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()

    ordered_classes = [0,2,1,3]
    import itertools
    plt.figure()
    cm=confusion_matrix(test_gold, np.argmax(test_preds,axis=1))
    plt.imshow(cm, cmap='Oranges')
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [seq_classes_rev[c] for c in range(num_classes)], rotation=45)
    plt.yticks(tick_marks, [seq_classes_rev[c] for c in range(num_classes)])
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if cfg['embed'] is not None:
        def plot_with_labels(low_dim_embs, labels, filename):
          assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
          plt.figure()  # in inches
          for i, label in enumerate(labels):
              if label is not '_':
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                plt.annotate(label,
                             xy=(x, y),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom',
                             fontsize='x-large')

          # plt.savefig(filename)

        try:
          # pylint: disable=g-import-not-at-top
          from sklearn.manifold import TSNE

          tsne = TSNE(perplexity=5, n_components=2, init='pca', n_iter=1000, method='exact')
          final_embeddings = sess.run(embedding_encoder)
          low_dim_embs = tsne.fit_transform(final_embeddings)
          labels = seq_tokens
          plot_with_labels(low_dim_embs, labels, os.path.join('./tsne.png'))

        except ImportError as ex:
          print('Please install sklearn, matplotlib, and scipy to show embeddings.')
          print(ex)

    # visualise representations of sequences
    # plt.figure()
    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000,early_exaggeration=2)
    # representations=[]
    # labels=[]
    # for j in range(len(train_data)//batch_size):
    #     batch_xs, _ = get_batch(train_data, j, batch_size)
    #     this_rep = sess.run(rnn_out, feed_dict={x:batch_xs})
    #     representations.extend(this_rep.tolist())
    #     labels.extend(batch_ys)
    # rep_arr = np.asarray(representations)
    # print(rep_arr.shape)
    # low_dim_reps = tsne.fit_transform(rep_arr)
    # colors=['red','blue','green','orange']
    # # print(low_dim_reps)
    # for i,label in enumerate(labels):
    #     plt.scatter(x=low_dim_reps[i,0],y=low_dim_reps[i,1], c=colors[label],s=1)

    plt.show()
