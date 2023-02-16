import pickle
import pandas as pd
li=pickle.load(open("info_dev.pickle","rb"))
sum_list=[]
for i in range(len(li)):
        a_sum=[]
        p=li[i]
        b=0
        for j in range(len(p)):
            b=b+p[j]
        sum_list.append(b)
    
print("lenght of development pickle",len(sum_list))

df=pd.DataFrame(sum_list)
print("Distributions of Development data set")
df.describe()
li=pickle.load(open("info_train.pickle","rb"))
train_list=[]
for i in range(len(li)):
        a_sum=[]
        p=li[i]
        b=0
        for j in range(len(p)):
            b=b+p[j]
        sum_list.append(b)
        train_list.append(b)
    
print("lenght of train pickle",len(train_list))

df=pd.DataFrame(train_list)
print("Distributions of Development data set")
df.describe()

li=pickle.load(open("info_train.pickle","rb"))
test_list=[]
for i in range(len(li)):
        a_sum=[]
        p=li[i]
        b=0
        for j in range(len(p)):
            b=b+p[j]
        sum_list.append(b)
        test_list.append(b)
    
print("lenght of test pickle",len(sum_list))

df=pd.DataFrame(test_list)
print("number of sentence Distributions of test data set")
df.describe()

df=pd.DataFrame(sum_list)
print("Global train test and dev  number of sentence Distributions of test data set")
df.describe()







