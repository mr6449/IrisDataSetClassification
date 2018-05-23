import numpy as np
import tflearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)


from tflearn.data_utils import load_csv
To_ignore=[2,3]
data , labels = load_csv('/home/meet/Downloads/iris.csv',target_column=4,columns_to_ignore=To_ignore,categorical_labels=True,n_classes=3)

net = tflearn.input_data(shape=[None,2])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=4, show_metric=True,validation_set=0.1)

k1=[4.2,2.7]
k2=[5.1,3.5]
k3=[4.7,3.2]
k4=[5.8,4]
k5=[4.9,2.4]
k6=[5.2,2.7]
k7=[5,2]
k8=[5.1,3.8]
k9=[4.2,3.4]
k10=[4.1,3.1]
arr = [k1,k2,k3,k4,k5,k6,k7,k8,k9,k10]
i=0
Arr1=[]
for i in np.arange(0,10):
	pred=model.predict(arr);
	print("Setosa",pred[i][0])
	print("Versicolor",pred[i][1])
	print("Virginica",pred[i][2])
	m=max(pred[i][0],pred[i][1],pred[i][2])
	if(m==pred[i][0]):
		print("Setosa")
		Arr1=np.append(Arr1,'0')
	elif(m==pred[i][1]):
		print("Versicolor")
		Arr1=np.append(Arr1,'1')
	else:
		print("Virginica")
		Arr1=np.append(Arr1,'2')
	i=i+1

	print("-------")
print(Arr1)

df=pd.read_csv('/home/meet/Downloads/iris.csv')
df1=pd.read_csv('/home/meet/iris2.csv')
df2=pd.read_csv('/home/meet/Downloads/iris3.csv')
df2=df
print(df1)
df1['variety1']=Arr1
print(df1)
#print(df.groupby('variety').size())
#df.boxplot(by="variety",figsize=(10,10))
#plt.show()
#sns.pairplot(df, hue="variety",diag_kind="kde")
#plt.show()

x1=[4.2,5.1,4.7,5.8,4.9,5.2,5,5.1,4.2,4.1]
y1=[2.7,3.5,3.2,4,2.4,2.7,2,3.8,3.4,3.1]
x=df["sepal.length"]
y=df["sepal.width"]
#x1,y1,x2,y2,x3,y3 = [],[],[],[],[],[]
i=0
while i<len(df):
	if(int(df.loc[i,'variety'])==1):
		plt.scatter(x[i],y[i],c='red',marker='x')
		
	if(int(df.loc[i,'variety'])==0):
		plt.scatter(x[i],y[i],c='blue',marker='x')
		
	if(int(df.loc[i,'variety'])==2):
		plt.scatter(x[i],y[i],c='yellow',marker='x')
	i = i+1
		
plt.xlabel('x')
plt.ylabel('y')	
plt.legend()

i=0
while i<len(x1):
	if(int(df1.loc[i,'variety1'])==1):
		plt.scatter(x1[i],y1[i],c='green')
		
	if(int(df1.loc[i,'variety1'])==0):
		plt.scatter(x1[i],y1[i],c='magenta')
		
	if(int(df1.loc[i,'variety1'])==2):
		plt.scatter(x1[i],y1[i],c='black')
	i = i+1
		
x2=df2["sepal.length"]
y2=df2["sepal.width"]
i=0
j=0
Arr3=[]
while i<len(df2):
	a1=[x2[i],y2[i]]
	Arry2=[a1]
	print(type(a1))
	pred=model.predict(Arry2)
	print("Start Pred")	
	print(pred)
	print("Setosa",pred[j][0])
	print("Versicolor",pred[j][1])
	print("Virginica",pred[j][2])
	m=max(pred[j][0],pred[j][1],pred[j][2])
	if(m==pred[j][0]):
		print("Setosa")
		Arr3=np.append(Arr3,'0')
	elif(m==pred[j][1]):
		print("Versicolor")
		Arr3=np.append(Arr3,'1')
	else:
		print("Virginica")
		Arr3=np.append(Arr3,'2')
	i=i+1

	print("-------")
print(Arr3)
df2['variety2']=Arr3
print(df2)
i=0
while i<len(df2):
	if(int(df2.loc[i,'variety2'])==1):
		plt.scatter(x2[i],y2[i],c='green')
	
	if(int(df2.loc[i,'variety2'])==0):
		plt.scatter(x2[i],y2[i],c='magenta')
	
	if(int(df2.loc[i,'variety2'])==2):
		plt.scatter(x2[i],y2[i],c='cyan')

	i=i+1


plt.xlabel('x')
plt.ylabel('y')	
plt.legend()
plt.show()
