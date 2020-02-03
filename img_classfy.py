
# coding: utf-8

# In[3]:


from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split


# In[4]:


# 분류할 대상의 카테고리와 디렉터리 경로를 설정합니다. 
glasses_dir = r"C:\Users\KKIM\Desktop"
categories = ["안경을 안 쓴 성인남자","안경을 쓴 성인남자","남자 아이","안경을 안 쓴 성인여자","안경을 쓴 성인여자","여자 아이"]
nb_classes = len(categories)


# In[6]:


#이미지의 크기를 지정합니다.  
image_w = 64 
image_h = 64
pixels = image_w * image_h * 3


# In[7]:


# 이미지를 디렉터리에서 가져와 읽습니다.    
x = []
y = []
for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]  
    label[idx] = 1  # 각 이미지에 카테고리별로 레이블을 지정해줍니다. 
    
    image_dir = glasses_dir + "\\" + str(idx+1)  #카테고리별로 1~6번 디렉터리안에 있는 이미지에 접근 경로를 설정합니다.  
    files = glob.glob(image_dir+ "\\" + "*.png") #이미지 디렉터리에 있는 .png파일들을 전부 가져와 리스트로 만듭니다.
    for i, f in enumerate(files):
        img = Image.open(f) 
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)      # numpy 배열로 변환
        x.append(data)
        y.append(label)


# In[23]:


x = np.array(x)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x,y) #테스트데이터와 학습데이터를 분리합니다.
xy = (x_train, x_test, y_train, y_test)
np.save(r"C:\Users\KKIM\Desktop/6obj.npy", xy)
print("ok,", len(y))


# In[24]:


from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

nb_classes = len(categories)

x_train, x_test, y_train, y_test = np.load(r"C:\Users\KKIM\Desktop/6obj.npy", allow_pickle=True)
# 데이터 정규화하기(0~1사이로)
x_train = x_train.astype("float") / 256
x_test  = x_test.astype("float")  / 256
print('x_train shape:', x_train.shape)

# 모델 구조 정의 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], padding='same')) #input_shape = (64,64,3)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Fully conected layer
model.add(Flatten())   
model.add(Dense(512))   
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])
# 모델 확인
#print(model.summary())

# 모델 훈련하기
#model.fit(x_train, y_train, batch_size=32, nb_epoch=20)

# 학습 완료된 모델 저장
hdf5_file = r"C:\Users\KKIM\Desktop/6obj-model.hdf5"
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    model.fit(X_train, y_train, batch_size=32, nb_epoch=10)
    model.save_weights(hdf5_file)


# In[25]:


score = model.evaluate(x_test, y_test)
print('loss=', score[0])        #손실값 출력
print('accuracy=', score[1])    #정확도 출력 

