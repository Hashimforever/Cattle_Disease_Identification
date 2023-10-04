
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3

class my_InceptionV3:
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.IMAGE_SIZE = [224, 224]
        self.inception = InceptionV3(input_shape=self.IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        self._freeze_layers()
        self._create_model()
        
    def _freeze_layers(self):
        for layer in self.inception.layers:
            layer.trainable = False
    
    def _create_model(self):
        x = Flatten()(self.inception.output)
        prediction = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=self.inception.input, outputs=prediction)
    
    def summary(self):
        self.model.summary()
model=my_InceptionV3(num_classes=5)
#print("this model is Pre_ResNet18Model1")
#model.summary()
