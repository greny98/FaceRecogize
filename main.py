from one_shot_model import siamese
from libs import *
import data_generator
import argparse

if __name__ == '__main__':
    # Handle arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.000005)
    args = vars(parser.parse_args())
    epochs = args['epochs']
    lr = args['lr']

    # Read data.csv
    df = pd.read_csv('data/data.csv')

    ds_train, ds_test = data_generator.create_ds(df)
    input1 = L.Input(shape=(300, 300, 3,))
    input2 = L.Input(shape=(300, 300, 3,))
    output = siamese(input1, input2)
    model = models.Model(inputs=[input1, input2], outputs=output)
    model.summary()
    model.compile(
        loss=losses.binary_crossentropy,
        optimizer=optimizers.Adam(learning_rate=lr),
        metrics=['acc'])
    model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=epochs
    )
    model.save('model.h5')
