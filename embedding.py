from libs import *
from siamese import SiameseModel, create_model
import data_generator


network, embedding = create_model()
siamese_model = SiameseModel(network)
for label in os.listdir('data/faces_test'):
    print(label)
    if label == '.DS_Store':
        continue
    label_path = os.path.join('data/faces_test', label)
    embs = []
    for image in os.listdir(label_path):
        if image == '.DS_Store':
            continue
        img = data_generator.preprocessing_image(os.path.join(label_path, image))
        img = tf.expand_dims(img, axis=0)
        emb = embedding(img)
        embs.append(emb.numpy()[0])
    df = pd.DataFrame(embs)
    df.to_csv(f'data/emb/{label}.csv', index=False)
