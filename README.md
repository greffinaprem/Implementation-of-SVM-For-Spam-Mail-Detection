# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program. 


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:GREFFINA SANCHEZ P 
RegisterNumber:212222040048  
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

1. Result output

   ![image](https://github.com/greffinaprem/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475603/966358c1-8738-434b-8b97-d6a3ae93bbd4)

3. data.head()

   ![image](https://github.com/greffinaprem/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475603/b7a932fc-1381-4039-b7c4-e0623de1a33d)

5. data.info()

   ![image](https://github.com/greffinaprem/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475603/acdf1d37-7be7-4c22-8873-2ac432884b38)

7. data.isnull().sum()

   ![image](https://github.com/greffinaprem/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475603/e1825090-1831-4296-b6df-9565df2bbd43)

9. Y_prediction value

    ![image](https://github.com/greffinaprem/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475603/9dc7cc8e-a8ff-4f49-9be0-ee8cbfe54a16)

6. Accuracy value
    
  ![image](https://github.com/greffinaprem/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119475603/ccc339e9-050c-4c76-aa27-d4f3eba2aa5d)


    



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
