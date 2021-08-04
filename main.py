from one_shot_model import siamese, encoding
from libs import *
import data_generator

if __name__ == '__main__':
    df = pd.read_csv('data/data.csv')
    ds_train, ds_test = data_generator.create_ds(df)
    input1 = L.Input(shape=(224, 224, 3,))
    input2 = L.Input(shape=(224, 224, 3,))
    output = siamese(input1, input2)
    model = models.Model(inputs=[input1, input2], outputs=output)
    model.summary()
    model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['acc'])
    model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=3
    )
