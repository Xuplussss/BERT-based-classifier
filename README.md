# BERT-based-classifier - [paper](<https://ieeexplore.ieee.org/abstract/document/9306243> "Title")


## Requirements
```
- pytorch_pretrained_bert
- scikit-learn
- imblearn
```

## 簡介
```
在許多情況，我們需要強大語言模型為我們壓縮並找出文字中的特徵。BERT是一個容易移植且已經過大量預訓練的模型。

本研究使用BERT對醫院紀錄之主訴來區分患者是否具有類流感風險。使用BERT作為實驗中的一項先進模型來分析結果。
```
### 資料範例
```
Sample:病患來診為眼睛紅癢 , 急性周邊中度疼痛(4 ~ 7)；左眼今天癢，刺痛感存求診，無分泌物
Ans: 非類流感
```
## 快速使用
```
bash run_bert_linear.sh
```

## 研究分析
我們使用tfidf計算分類類流感的重要字詞，並簡易的使用keyword match方法與其他word embedding方法進行比較

!["Table 3 "](https://github.com/Xuplussss/BERT-based-classifier/blob/main/table.PNG?raw=true)

使用keyword match方法已能準確地預測類流感，但BERT模型能帶來突破性的提升。


!["Fig 1"](https://github.com/Xuplussss/BERT-based-classifier/blob/main/Fig1.PNG?raw=true)

由上圖可看出，即使各方法得到的預測取線起伏相似，BERT模型在一年中的判斷能更與臨床吻合，對於冬季高峰也預測的較準。

## Reference
This package provides training code for the [ILI prediction paper](<https://ieeexplore.ieee.org/abstract/document/9306243> "Title"). If you use this codebase in your experiments please cite: 

```
@inproceedings{hsu2020natural,
  title={Natural Language Processing Methods for Detection of Influenza-Like Illness from Chief Complaints},
  author={Hsu, Jia-Hao and Weng, Ting-Chia and Wu, Chung-Hsien and Ho, Tzong-Shiann},
  booktitle={2020 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  pages={1626--1630},
  year={2020},
  organization={IEEE}
}
```
