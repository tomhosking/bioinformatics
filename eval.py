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

embedding_size=8
num_conv_filters=None

x,y,opt, accuracy, y_hat, loss, embedding_encoder,dropout_active = build_graph(num_conv_filters=num_conv_filters,embedding_size=embedding_size)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
    saver = tf.train.Saver()

    saver.restore(sess, './model/model.checkpoint')
    best_dev_acc=0
    print('Final model evaluation:')

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
        batch_xs, batch_ys = get_batch(dev_data, j, batch_size)
        dev_acc,dev_pred= sess.run([accuracy,y_hat], feed_dict={x:batch_xs, y:batch_ys})
        dev_accs.append(dev_acc)
        dev_gold.extend(batch_ys)
        # print(dev_pred.tolist())
        # exit()
        dev_preds.extend(dev_pred.tolist())
    print('Dev: ', np.mean(dev_accs) )

    # Only check the test set after model selection!
    test_accs =[]
    for j in range(len(test_data)//batch_size):
        batch_xs, batch_ys = get_batch(test_data, j, batch_size)
        test_acc,test_pred= sess.run([accuracy,y_hat], feed_dict={x:batch_xs, y:batch_ys})
        test_accs.append(test_acc)
    print('Test: ' ,np.mean(test_accs) )

    dev_preds=np.asarray(dev_preds)


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(np.equal(dev_gold,i), dev_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    for c in range(num_classes):
        plt.plot(fpr[c], tpr[c],
             lw=lw, label='{:} (area = {:0.2f})'.format( seq_classes_rev[c], roc_auc[c]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()

    import itertools
    plt.figure()
    cm=confusion_matrix(dev_gold, np.argmax(dev_preds,axis=1))
    plt.imshow(cm)
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

    if embedding_size is not None:
        def plot_with_labels(low_dim_embs, labels, filename):
          assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
          plt.figure()  # in inches
          for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

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


    plt.show()
