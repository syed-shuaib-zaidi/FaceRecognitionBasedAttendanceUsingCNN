import cv2
import pandas as pd
import numpy as np
from tkinter import messagebox
import tkinter as tk
import os
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
import pickle
from PIL import Image
import shutil

detector = MTCNN()
def register(txt,txt2):
	t = tk.Tk()
	t.geometry('+1050+120')
	t.configure(background='#122c57')
	l1 = tk.Label(t,text="Taking 100 Photos\n",fg='white',bg='#122c57')
	l1.pack()
	#Init Camera
	cap = cv2.VideoCapture(0)

	skip = 0
	face_data = []
	name = txt2.get().upper()
	roll_no = txt.get().upper()

	l2 = tk.Label(t,text=str(0)+"\n",fg='white',bg='#122c57')
	l2.pack()			
	t.update()

	while True:
		ret,frame = cap.read()

		if ret==False:
			continue

		faces = detector.detect_faces(frame)
		if len(faces)==0:
			print('Your face is not visible, please get into the frame\n')
			continue
			
		x, y, w, h  = faces[0]['box']
		offset = 10
		cv2.rectangle(frame,(x-offset,y-offset),(x+w+offset,y+h+offset),(0,255,255),2)
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(224,224))
		face_section = cv2.cvtColor(face_section,cv2.COLOR_BGR2RGB)
		face_data.append(face_section)
		print(len(face_data))
		skip += 1
		if skip%10 == 0:

				l2 = tk.Label(t,text=str(len(face_data))+"\n",fg='white',bg='#122c57')
				l2.pack()			
				t.update()	
		

		cv2.imshow("FRAME",frame)
		cv2.imshow("FACE",face_section)
		key_pressed = cv2.waitKey(1) & 0xFF
		if key_pressed == ord('q') or len(face_data) >= 100:
			t.destroy()
			break

	cap.release()
	cv2.destroyAllWindows()

	# Save this data into file system
	train_path='./train/'
	if os.path.isdir('{}{}'.format(train_path,roll_no)):
		shutil.rmtree('{}{}'.format(train_path,roll_no))
	os.mkdir('{}{}'.format(train_path,roll_no))
	print('New directory {} created at train folder'.format(roll_no))
	cnt=1
	for i in range(len(face_data)-10):
		face = face_data[i]
		img=Image.fromarray(face)  
		path = '{}{}/{}.jpeg'.format(train_path,roll_no,cnt)
		cnt+=1
		img.save(path)
	print("Training Data Successfully save at "+train_path+roll_no)

	validate_path='./validate/'
	if os.path.isdir('{}{}'.format(validate_path,roll_no)):
		shutil.rmtree('{}{}'.format(validate_path,roll_no))
	os.mkdir('{}{}'.format(validate_path,roll_no))
	print('New directory {} created at validate folder'.format(roll_no))
	cnt=1
	for i in range(10):
		face = face_data[len(face_data)-i-1]
		img=Image.fromarray(face)  
		path = '{}{}/{}.jpeg'.format(validate_path,roll_no,cnt)
		cnt+=1
		img.save(path)
	print("Validation Data Successfully save at "+validate_path+roll_no)

	# Registering student in csv file
	row = np.array([roll_no,name]).reshape((1,2))
	df = pd.DataFrame(row) 
	# if file does not exist write header
	if not os.path.isfile('student_details.csv'):
	   df.to_csv('student_details.csv', header=['roll','name'],index=False)
	else: # else it exists so append without writing the header
	   df.to_csv('student_details.csv', mode='a', header=False,index=False)
		
	
	tk.messagebox.showinfo("Notification", "You have been registered successfully") 
