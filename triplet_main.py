from siamese import SiameseModel, create_model
from libs import *
import data_generator
import argparse

if __name__ == '__main__':
    # Handle arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = vars(parser.parse_args())
    epochs = args['epochs']
    lr = args['lr']

    # Read data.csv
    df = pd.read_csv('data/triplet_data.csv')

    ds_train, ds_test = data_generator.create_ds_triplet(df)
    network = create_model()
    siamese_model = SiameseModel(network)
    siamese_model.compile(optimizer=optimizers.Adam(learning_rate=lr))
    siamese_model.fit(ds_train, epochs=epochs, validation_data=ds_test)
    siamese_model.save('model.h5')
# df = data_generator.create_df_triplet('data/faces', 5000)
# df.to_csv('data/triplet.csv', index=False)
