from libs import *
from one_shot_model import siamese
import data_generator

if __name__ == '__main__':
    input1 = L.Input(shape=(300, 300, 3,))
    input2 = L.Input(shape=(300, 300, 3,))
    output = siamese(input1, input2)
    model = models.Model(inputs=[input1, input2], outputs=output)
    model.load_weights('models/model.h5')
    img1 = data_generator.preprocessing_image('data/faces/Dang Thi Tuyen/1_0.jpeg')
    img1 = tf.expand_dims(img1, axis=0)
    img2 = data_generator.preprocessing_image('data/faces/Tra My/1_0.jpeg')
    img2 = tf.expand_dims(img2, axis=0)
    results = model.predict((img1, img2))
    print(results)
