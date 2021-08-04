from libs import *
import random
import cv2


def get_random(images, used):
    existed = True
    idx = 0
    c = 0
    while existed:
        idx = random.randint(0, len(images) - 1)
        if images[idx].endswith('.DS_Store'):
            continue
        existed = used.get(images[idx], False)
        if c >= 8:
            break
        c += 1
    return images[idx]


def create_df(image_dir):
    used = {}
    n_pos = 50
    n_neg = 16
    dataset = []
    # positive
    for label in os.listdir(image_dir):
        dir_path = os.path.join(image_dir, label)
        images = [os.path.join(dir_path, filename)
                  for filename in os.listdir(dir_path)]
        for i in range(n_pos):
            img1 = get_random(images, used)
            used[img1] = True
            img2 = get_random(images, used)
            used[img2] = True
            dataset.append([img1, img2, 1])
            print([img1, img2, 1])
    # negative
    neg_pair = product(
        [os.path.join(image_dir, label) for label in os.listdir(image_dir)],
        repeat=2
    )
    print(neg_pair)
    for label1, label2 in neg_pair:
        if label1 == label2:
            continue
        images1 = [os.path.join(label1, filename)
                   for filename in os.listdir(label1)]
        images2 = [os.path.join(label2, filename)
                   for filename in os.listdir(label2)]
        for i in range(n_neg):
            img1 = get_random(images1, used)
            used[img1] = True
            img2 = get_random(images2, used)
            used[img2] = True
            dataset.append([img1, img2, 0])
            print([img1, img2, 0])
    return pd.DataFrame(dataset, columns=['img1', 'img2', 'same'])


def preprocessing_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, size=(224, 224))
    img = img / 127.5 - 1.
    return img


def create_ds(df: pd.DataFrame):
    imgs1 = df['img1'].values
    imgs2 = df['img2'].values
    same = df['same'].values

    def load_data(imgs, is_same):
        img1 = preprocessing_image(imgs[0])
        img2 = preprocessing_image(imgs[1])
        return (img1, img2), is_same

    n_samples = same.shape[0]
    n_train = int(0.85 * n_samples)
    # tensor slices
    imgs1_tensor = tf.data.Dataset.from_tensor_slices(imgs1)
    imgs2_tensor = tf.data.Dataset.from_tensor_slices(imgs2)

    image_ds = tf.data.Dataset.zip((imgs1_tensor, imgs2_tensor))
    labels = tf.data.Dataset.from_tensor_slices(same)
    ds = tf.data.Dataset.zip((image_ds, labels))
    ds = ds.shuffle(128)
    ds = ds.map(load_data)
    ds_train = ds.take(n_train)
    ds_test = ds.skip(n_train)
    ds_train = ds_train.shuffle(128, reshuffle_each_iteration=True) \
        .batch(32, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(32, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test


# def create_df_triplet():
