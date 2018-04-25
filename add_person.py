""" Этот скрипт служит для добавление нового юзера в систему распознаваний!"""
import glob
import os
import cv2
# import numpy
import config
# import face
import select
import sys
import face_recognition

name = input('Enter name: ')
surname = input('Enter surname: ')

user_folder_prefix = name.capitalize() + surname.capitalize()


def is_letter_input(letter):
    """Для проверки введенного через консоль символа"""
    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        input_char = sys.stdin.read(1)
        return input_char.lower() == letter.lower()
    return False


if __name__ == '__main__':

    counter = 0

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if not os.path.exists(config.TRAINING_DIR + user_folder_prefix):
        os.makedirs(config.TRAINING_DIR + user_folder_prefix)

    files = sorted(glob.glob(os.path.join(config.TRAINING_DIR + user_folder_prefix, name + '[0-9][0-9].jpg')))

    if len(files) > 0:
        counter = int(files[-1][-7:-4]) + 1

    print('Capturing images for user: {0} {1} !'.format(name.capitalize(), surname.capitalize()))
    print('Press type "c" and press "Enter/Return" to take a photo!')
    print('Press "Ctrl-C" to quit!')

    while True:

        ret, frame = cap.read()

        if not ret:
            print('Camera error!')

        cropped_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        amount_of_faces = face_recognition.face_locations(cropped_frame)

        if len(amount_of_faces) > 1:
            cv2.putText(frame, 'please ensure there is only one person in the picture!', (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            print('please ensure there is only one person in the picture!')
        elif len(amount_of_faces) == 1:
            cv2.putText(frame, 'please, press key "c" to take photo!', (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            if cv2.waitKey(1) & 0xFF == ord('c') or is_letter_input('c'):
                file_name = os.path.join(config.TRAINING_DIR + user_folder_prefix, name + '%03d.jpg' % counter)
                cv2.imwrite(file_name, frame)

                print('Found face and wrote training image', file_name)
                cv2.putText(frame, 'Found face and wrote training image {0}'.format(file_name), (60, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                counter += 1

        else:
            cv2.putText(frame, 'no face detected!', (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            print('No face detected!')

        cv2.putText(frame, 'Captured photos: {0}'.format(counter), (300, 300), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0),
                    2)
        cv2.imshow('New person frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

"""
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = face.detect_faces(gray)

		
		#for (x, y, w, h) in faces:
		#		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)



		if cv2.waitKey(1) & 0xFF == ord('c') or is_letter_input('c'):

			result = face.detect_single_face(gray)
			
			if faces is None:
				cv2.putText(frame, 'Please ensure there is only one person in the picture!', (0, 20),
                  	  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
#						print('Please ensure there is only one person in the picture!')
				


			x, y, w, h = result

			file_name = os.path.join(config.TRAINING_DIR + user_folder_prefix, name + '%03d.jpg' % counter)

			cv2.imwrite(file_name, frame)

			print('Found face and wrote training image', file_name)
			cv2.putText(frame, 'Found face and wrote training image {0}'.format(file_name), (0, 20),
                  	  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

			counter += 1

		cv2.imshow('Training frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
"""
