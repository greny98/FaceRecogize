from libs import *

pre_trained = App.MobileNetV2(
    input_shape=(224, 224, 3,),
    weights='imagenet',
    include_top=False
)

for l in pre_trained.layers:
    l.trainable = False


def cnn_feature_extractor(imgs):
    """
    Use Mobile Net to extract features
    :return:
    """
    cnn_feature = pre_trained(imgs)
    return L.GlobalAveragePooling2D()(cnn_feature)


def encoding(imgs, n_units=256):
    cnn_feature = cnn_feature_extractor(imgs)
    enc_feature = L.Dense(
        n_units,
        kernel_initializer=initializers.HeNormal(),
    )(cnn_feature)
    norm = tf.linalg.l2_normalize(enc_feature, axis=-1, name='encoding')
    return norm


def siamese(img1, img2, units=256):
    vec1 = encoding(img1, units)
    vec2 = encoding(img2, units)
    vec = L.Subtract()([vec1, vec2])
    DistanceLayer = Lambda(lambda tensors: tf.abs(tensors))
    dist = DistanceLayer(vec)
    outputs = L.Dense(1, activation='sigmoid',
                      kernel_initializer=initializers.HeNormal())(dist)
    return outputs
