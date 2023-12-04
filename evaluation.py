import pickle, argparse
import numpy as np 
from sklearn.metrics import roc_curve, roc_auc_score
from math import sqrt

def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return lower, upper
def statistic(cm):
    total = sum(sum(cm))
    sen = cm[1,1]/(cm[1,0]+cm[1,1])
    spe = cm[0,0]/(cm[0,0]+cm[0,1])
    if spe == 0:
        spe = 0.0001
    PPV = cm[1,1]/(cm[1,1]+cm[0,1])
    AA = (cm[0,0]+cm[1,0]) 
    if AA == 0:
        AA = 0.0001
    NPV = cm[0,0]/AA
    FPR = 1 - spe
    if FPR == 0:
        FPR = 0.0001
    FNR = 1 - sen
    if FNR == 0:
        FNR = 0.0001
    LR_p = sen / FPR
    LR_m = FNR / spe
    acc = (cm[0,0]+cm[1,1])/total
    return [acc, sen, spe, PPV, NPV, LR_p, LR_m]
acc = []
sen = []
spe = []
PPV = []
NPV = []
LR_p = []
LR_m = []


for fold_num in range(10):
    oup = []
    test_answer_file = open('data/tenfold/fold'+str(fold_num)+'/test_answer.txt', 'r').readlines()
    for line in test_answer_file:
        oup.append(int(line))
    oup = np.asarray(oup)

    prediction_proba = np.load('bert_result/chief/fold'+str(fold_num)+'/output_profile.npy')
    threshold = 0.9
    y_pred_new_threshold = (prediction_proba[:,1]>=threshold).astype(int)
    cm = np.zeros([2,2])
    for i in range(len(y_pred_new_threshold)):
        cm[oup[i], y_pred_new_threshold[i]] += 1
    result = statistic(cm)
    threshold -= 0.0001
    while result[1] < 0.85:
        y_pred_new_threshold = (prediction_proba[:,1]>=threshold).astype(int)
        cm = np.zeros([2,2])
        for i in range(len(y_pred_new_threshold)):
            cm[oup[i], y_pred_new_threshold[i]] += 1
        result = statistic(cm)
        threshold -= 0.01
    print('fold', fold_num, threshold)
    print(result)
    print('---AUC:',roc_auc_score(oup, prediction_proba[:,1]))
    if result[1] <= 0.99:
        acc.append(result[0])
        sen.append(result[1])
        spe.append(result[2])
        PPV.append(result[3])
        NPV.append(result[4])
        LR_p.append(result[5])
        LR_m.append(result[6])
    
    if fold_num == 9:
        predictions = prediction_proba
        print('AUC',roc_auc_score(oup, predictions[:,1]))
        low,upper = roc_auc_ci(oup,predictions[:,1])
        # print(predictions[0],type(predictions[0]))
        print('AUC 95CI', low,upper)
        fpr, tpr, _ = roc_curve(oup, predictions[:,1])
        # plt.clf()
        # plt.plot(fpr, tpr)
        # plt.xlabel('FPR')
        # plt.ylabel('TPR')
        # plt.title('ROC curve')
        # plt.show()
acc = np.asarray(acc)
sen = np.asarray(sen)
spe = np.asarray(spe)
PPV = np.asarray(PPV)
NPV = np.asarray(NPV)
LR_p = np.asarray(LR_p)
LR_m = np.asarray(LR_m)
print('outside accuracy:', np.mean(acc), np.std(acc))
print('sen', np.mean(sen), np.std(sen))
print('spe', np.mean(spe), np.std(spe))
print('PPV', np.mean(PPV), np.std(PPV))
print('NPV', np.mean(NPV), np.std(NPV))
print('LR+', np.mean(LR_p), np.std(LR_p))
print('LR-', np.mean(LR_m), np.std(LR_m))
