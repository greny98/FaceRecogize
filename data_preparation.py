import time
import cv2
import os


def face_detection(img, classifier):
    # img = cv2.pyrDown(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, ksize=(3, 3))
    boxes = classifier.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6)
    results = []
    for (x, y, w, h) in boxes:
        results.append(img[y:y + h, x:x + w, :])
    return results, boxes


def crop_face_from_video(vid_file, save_dir, classifier, skip):
    print(f'Video {vid_file}')
    cap = cv2.VideoCapture(vid_file)
    step = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # If end => break
        if not ret:
            break
        # Skip if step < skip value
        if step != 0 and step < skip:
            step += 1
            continue
        # Reset step when step == skip and skip different 0
        elif step == skip and skip != 0:
            step = 0
            continue
        faces = face_detection(frame, classifier)
        path_dir = f'data/{save_dir}'
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        for idx, face in enumerate(faces):
            img_path = f'{path_dir}/{frame_idx}_{idx}.jpeg'
            cv2.imwrite(img_path, face)
        frame_idx += 1


def handle_vid_dir(vid_dir, classifier, skip=2):
    for idx, vid in enumerate(os.listdir(vid_dir)):
        print("+++", os.path.join(vid_dir, vid))
        crop_face_from_video(os.path.join(vid_dir, vid),
                             f'u_{idx}', classifier, skip)
        time.sleep(3.)
