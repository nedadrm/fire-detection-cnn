import numpy as np
import cv2
import os #operating system
from sklearn.model_selection import train_test_split #750eğitim 253 test 
from keras.models import Sequential #katman sırası
from keras.layers import Conv2D,Dense,Dropout,MaxPooling2D,Flatten
from keras.utils import to_categorical


egitim_dataset = ["fire_images",
 "non_fire_images"]

def load_images(egitim_dataset) :
    images = []
    labels = []


    for i in range(len (egitim_dataset)) :
        folder = egitim_dataset[i] 
        label = i
        #print(folder)
        for filename in os.listdir(folder) :#listedeki resimleri tek tek al #dongu icinde dongu kurma
            try:
             #print(folder, filename)
             img = cv2.imread(os.path.join(folder, filename))
             img = cv2.resize(img, (48, 48))
             images.append(img)
             labels.append(label)
            except Exception as e:
               print(f"resim okunamadı{os.path.join(folder, filename)} : {e}")  #neden sonra e var

    return np.array (images), np.array (labels) #resimleri degere donusturdu
    
images, labels = load_images(egitim_dataset)
print(len(images))
print(len(labels))
#resim ve değerlerin yeniden boyutlandırılması
X_train, X_test, y_train, y_test = train_test_split(images,labels,test_size=0.2,random_state=42)
X_train = X_train.reshape (X_train.shape[0], 48, 48, 3).astype('float32')/255
X_test = X_test.reshape(X_test.shape [0], 48, 48, 3).astype('float32')/255

Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = Sequential()
model.add(Conv2D (32, kernel_size= (3,3), activation="relu",  input_shape=(48,48,3)))#resimlerden gerekli özelliklerin çıakrılması (verilein bölünerek incelenmesi)
model.add(Conv2D (32, kernel_size= (3,3), activation="relu"))
model.add(MaxPooling2D (pool_size= (2,2)))  #boyut küçültme
model.add(Dropout (0.25)) # modelin öğrenirken genelleme yapmaması için bazı işlevleri engeller
model.add(Flatten()) #verinin uygun forma getirilmesiş
model.add(Dense (128, activation= "relu")) # özellikleri yorumlama
model.add(Dropout (0.5)) 
model.add(Dense (2, activation= "softmax")) #yangın var mı yok mu


model.compile(loss=('categorical_crossentropy'), optimizer='adam', metrics = ['accuracy']) #optimizer hatayı minimslize eder
#compile modelin nasıl ogrencegini belirtir

model.fit(X_train,  Y_train, batch_size=64 ,epochs=10, verbose=1, validation_data=(X_test , Y_test))

"""fit ise verilerle eigitlme asamasi
epochs= veriler kç kez modelden gececek 
verbose: egitimi yazdırma


"""

model.save("fire_detection_model.h5")


