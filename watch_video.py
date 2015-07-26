import cv2
import glob, os
os.chdir("offices_part2\office_0015")
for file in glob.glob("*.ppm"):
	# print(file)
	frame = cv2.imread(file)
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
print "terminado"