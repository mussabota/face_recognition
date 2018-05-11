from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import face_recognition
import face
import config
from datetime import datetime


modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./training_dir"

person_name = os.listdir(config.CHECK_FACE_FOLDER)
known_encodings = []
names = []

for p_name in person_name:
    person = face_recognition.load_image_file(os.path.join(config.CHECK_FACE_FOLDER, p_name))
    face_encoding = face_recognition.face_encodings(person)[0]
    known_encodings.append(face_encoding)
    names.append(os.path.splitext(p_name))


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        print('Loading Modal')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        # video_capture = cv2.VideoCapture(config.VIDEO_SOURCE)
        video_capture = config.capturing()

        c = 0

        print('Start Recognition')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

            curTime = time.time() + 1  # calc fps
            timeF = frame_interval

            if not ret:
                print('Camera error!')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face.detect_faces(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)


            cropped_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
            amount_of_faces = face_recognition.face_locations(cropped_frame)


            if len(amount_of_faces) > 1:
                cv2.putText(frame, 'please ensure there is only one person in the picture!', (0, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                #print('please ensure there is only one person in the picture!')

            elif len(amount_of_faces) == 1:
                cv2.putText(frame, 'please, press key "r" to recognize person!', (0, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                face_encoding = face_recognition.face_encodings(cropped_frame, amount_of_faces)[0]

                distance = face_recognition.face_distance(known_encodings, face_encoding)
                idx = np.argmin(distance)

                if distance[idx] < 0.5:

                    if (c % timeF == 0) and cv2.waitKey(1) & 0xFF == ord('r'):
                        find_results = []

                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('Detected_FaceNum: %d' % nrof_faces)

                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(frame.shape)[0:2]

                            cropped = []
                            scaled = []
                            scaled_reshape = []
                            bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                            for i in range(nrof_faces):
                                emb_array = np.zeros((1, embedding_size))

                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]

                                # inner exception
                                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(
                                        frame):
                                    print('Face is very close!')
                                    continue

                                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                       interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                print(predictions)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                # print("predictions")
                                # print(best_class_indices, ' with accuracy ', best_class_probabilities)
                                print("Prediction value is {0} %".format(int(best_class_probabilities * 100)))
                                # print(curTime)

                                #  print(best_class_probabilities)
                                if best_class_probabilities > 0.7:

                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                                                  2)  # boxing face

                                    if not config.door_status():
                                        config.set_status(True)


                                    # plot result idx under box
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    # print('Result Indices: ', best_class_indices[0])
                                    name, surname = str(HumanNames[best_class_indices[0]]).split('_')
                                    # print(name, surname)


                                    for H_i in HumanNames:
                                        if HumanNames[best_class_indices[0]] == H_i:
                                            result_names = name + ' ' + surname  # HumanNames[best_class_indices[0]]
                                            fixed_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' '

                                            print(result_names, 'last entered at ', fixed_time)
                                            cv2.putText(frame, result_names, (text_x, text_y),
                                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 255), thickness=1, lineType=2)
                        else:
                            print('Alignment Failure')

                else:
                    cv2.putText(frame, 'Sorry! Could not find any match!', (0, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if (c % timeF == 0) and cv2.waitKey(1) & 0xFF == ord('r'):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Detected_FaceNum: %d' % nrof_faces)

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('Face is very close!')
                                continue

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            print(predictions)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            # print("predictions")
                            # print(best_class_indices, ' with accuracy ', best_class_probabilities)
                            print("Prediction value is {0} %".format(int(best_class_probabilities * 100)))
                           # print(curTime)

                            #  print(best_class_probabilities)
                            if best_class_probabilities > 0.7:
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                                              2)  # boxing face

                                # plot result idx under box
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                # print('Result Indices: ', best_class_indices[0])
                                name, surname = str(HumanNames[best_class_indices[0]]).split('_')
                               # print(name, surname)
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = name + ' ' + surname  # HumanNames[best_class_indices[0]]
                                        if name == 'Nursultan':
                                           result_names = result_names + '- Ауыл спортын дамытушы'

                                        print(result_names)
                                        cv2.putText(frame, result_names, (text_x, text_y),
                                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 255), thickness=1, lineType=2)
                    else:
                        print('Alignment Failure')

            else:
                cv2.putText(frame, 'no face detected!', (0, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                #print('No face detected!')


            # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
