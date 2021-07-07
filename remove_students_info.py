import os
f = open('student_details.csv', "r+")
lines = f.readlines()
if len(lines) > 0:
	lines.pop()
f.close()
f = open('student_details.csv', "w+")
f.writelines(lines)
f.close()
os.remove('saved/id_to_roll_f.pkl')
os.remove('saved/roll_to_id_f.pkl')
f = open('saved/id_to_roll_f.pkl','w')
f.close()
f = open('saved/roll_to_id_f.pkl','w')
f.close()


