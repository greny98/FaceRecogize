from libs import *


def embedding():
    base_cnn = App.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(300, 300, 3))
    # freeze base_cnn
    trainable = False
    for layer in base_cnn.layers:
        layer.trainable = trainable
    # embedding
    flatten = L.GlobalAveragePooling2D()(base_cnn.output)
    dense1 = L.Dense(512, activation="relu")(flatten)
    dense1 = L.BatchNormalization()(dense1)
    dense2 = L.Dense(256, activation="relu")(dense1)
    dense2 = L.BatchNormalization()(dense2)
    output = L.Dense(256)(dense2)
    return models.Model(base_cnn.input, output, name="Embedding")


class DistanceLayer(L.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        print()
        return ap_distance, an_distance


def create_model(target_shape=(300, 300)):
    embedding_model = embedding()
    anchor_input = L.Input(name="anchor", shape=target_shape + (3,))
    positive_input = L.Input(name="positive", shape=target_shape + (3,))
    negative_input = L.Input(name="negative", shape=target_shape + (3,))
    distances = DistanceLayer()(
        embedding_model(anchor_input),
        embedding_model(positive_input),
        embedding_model(negative_input),
    )

    siamese_network = models.Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return siamese_network


class SiameseModel(models.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
