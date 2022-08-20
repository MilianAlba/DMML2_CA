import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ht_df = pd.read_csv('resultsIndoorModel.csv')
acc = ht_df['accuracy']
val_acc = ht_df['val_accuracy']

loss = ht_df['loss']
val_loss = ht_df['val_loss']
epochsc = ht_df['epochs']
precision = ht_df['precision']
val_prec = ht_df['val_precision']

recall = ht_df['recall']
val_recall = ht_df['val_recall']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)

plt.plot(epochsc, val_acc, label='Validation Accuracy')
plt.plot(epochsc, acc, label='Training Accuracy')
plt.ylim([0.6, 1])
plt.plot([10, 10],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochsc, loss, label='Training Loss')
plt.plot(epochsc, val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([10,10 ],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.ylabel('Cross Entropy')
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochsc,precision,  label='Training Precision')
plt.plot(epochsc, val_prec,  label='Validation Precision')
plt.ylim([0.8, 1])
plt.plot([10,10],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Precision')
plt.ylabel('Precision')

plt.subplot(2, 1, 2)
plt.plot(epochsc, recall, label='Training Recall')
plt.plot(epochsc, val_recall, label='Validation Recall')
plt.ylim([0, 1.0])
plt.plot([10, 10],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Recall')
plt.xlabel('epoch')
plt.ylabel('Recall')
plt.show()

results = pd.read_csv('testresults.csv')
results['Value'] =pd.Series(results['Value'], dtype='Float64')
resultsACC = results[results['Metric']=='accuracy ']
raind = resultsACC[results['Test set']=='indoor']
raout = resultsACC[results['Test set']=='outdoor']
models = list(raind['Model'])
indAcc= list(raind['Value'])
OutAcc= list(raout['Value'])

N = 3
ind = np.arange(N)
width = 0.35
plt.bar(ind, indAcc, width, label='Indoor Test', color='blue')
plt.bar(ind + width, OutAcc, width,
        label='Outdoor Test', color='green')

plt.ylabel('Accuracy')
plt.title('Accuracy by Model and Test subset')

plt.xticks(ind + width / 2, models)
plt.legend(loc='best')
plt.show()

resultsprec = results[results['Metric']=='precision ']
pind = resultsprec[results['Test set']=='indoor']
paout = resultsprec[results['Test set']=='outdoor']
indPrec= list(pind['Value'])
OutPrec= list(paout['Value'])

plt.bar(ind, indPrec, width, label='Indoor Test', color='grey')
plt.bar(ind + OutPrec, OutAcc, width,
        label='Outdoor Test', color='cyan')

plt.ylabel('Precision')
plt.title('Precision by Model and Test subset')

plt.xticks(ind + width / 2, models)
plt.legend(loc='best')
plt.show()