from libs import *
from siamese import SiameseModel, create_model
import data_generator
from data_preparation import face_detection
import cv2
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from tempfile import TemporaryFile
from playsound import playsound

haar_path = 'haar_features/haarcascade_frontalface_default.xml'
cosine_similarity = metrics.CosineSimilarity()


r = sr.Recognizer()


def text_to_speech(text, lang='vi'):
    tts = gTTS(text.lower(), lang=lang)
    filename = 'tmp.mp3'
    tts.save(filename)
    playsound(filename)
    os.remove(filename)


def load_embedding(model):
    main_features = {}
    for label in os.listdir('data/faces_test'):
        print("loading: ", label)
        if label == '.DS_Store':
            continue
        label_path = os.path.join('data/faces_test', label)
        embs = []
        for image in os.listdir(label_path):
            if image == '.DS_Store':
                continue
            img = data_generator.preprocessing_image(os.path.join(label_path, image))
            img = tf.expand_dims(img, axis=0)
            emb = model(img)
            embs.append(emb)
        main_features[label] = embs
    return main_features


def find_best(emb, main_features):
    emb = tf.convert_to_tensor(emb, dtype=tf.float32)
    best_candidate = ''
    best_score = -100
    for key, features in main_features.items():
        sims = []
        for feature in features:
            feature_tensor = tf.convert_to_tensor(feature, dtype=tf.float32)
            sim = cosine_similarity(emb, feature_tensor)
            sims.append(sim.numpy())
        sims = np.array(sims)
        avg_sim = np.average(sims)
        if best_score < avg_sim:
            best_candidate = key
            best_score = avg_sim

    print('best_score', best_score, best_candidate)
    return best_candidate, best_score


def embedding_face(face, model):
    tensor = data_generator.preprocessing_frame(face)
    emb = model(tensor)
    print("=====", emb.get_shape())
    # emb = emb.numpy()
    return emb


if __name__ == '__main__':
    # Load model
    network, embedding = create_model()
    siamese_model = SiameseModel(network)
    siamese_model.load_weights('models/ckpt/checkpoint')
    classifer = cv2.CascadeClassifier(haar_path)
    cap = cv2.VideoCapture(0)

    # Load main features
    main_features = load_embedding(embedding)
    # img1 = data_generator.preprocessing_image('data/faces_test/Nguyen Anh Tuan/5_0.jpeg')
    # emb1 = embedding(tf.expand_dims(img1, axis=0))
    # Videos
    while 1:
        ret, frame = cap.read()
        faces, boxes = face_detection(frame, classifer)
        n_faces = len(faces)
        names = []
        if n_faces != 0:
            embs = []
            for face in faces:
                embs.append(embedding_face(face, embedding))
            for idx, emb in enumerate(embs):
                cand, score = find_best(emb, main_features)
                print(cand, score)
                names.append(cand)
                # positive_similarity = cosine_similarity(emb1, emb)
                # print("Positive similarity:", positive_similarity.numpy())
        print(names)
        for idx, (x1, y1, w1, h1) in enumerate(boxes):
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (x1, y1)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            text_to_speech('Chào ' + names[idx])
            frame = cv2.putText(frame, names[idx], org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

# Load model
# network, embedding = create_model()
# siamese_model = SiameseModel(network)
# siamese_model.load_weights('models/ckpt/checkpoint')
# img1 = data_generator.preprocessing_image('data/faces/Dang Thi Tuyen/0_0.jpeg')
# emb1 = embedding(tf.expand_dims(img1, axis=0))
# img2 = data_generator.preprocessing_image('data/faces/Nguyen Anh Tuan/54_0.jpeg')
# emb2 = embedding(tf.expand_dims(img2, axis=0))
#
# positive_similarity = cosine_similarity(emb1, emb2)
# print("Positive similarity:", positive_similarity.numpy())
# print(emb1.shape, emb2.shape)
