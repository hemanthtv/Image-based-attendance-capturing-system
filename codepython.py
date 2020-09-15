import fnmatch
import face_recognition as fr
import os
from matplotlib import pyplot as plt
import cv2
Known_encoding_values = []
encodings_tobe_Check = []
matchinglist = []
i = 0
paths=r'C:\Users\Hemanth\Desktop\trainset\trainset'
for root,_,files in os.walk(paths):
    true=0
    false=0
    for filename in files: 
        matchinglist = []
        file = os.path.join(root,filename)
        if fnmatch.fnmatch(file,'*script.jpg'):
            label = file
            #print("label=",label)
            test = cv2.imread(label)        
            encodings_tobe_Check= fr.face_encodings(test)[0]
                    
        elif fnmatch.fnmatch(file,"*.jpg"):
            image = file
            #print("image=",image)
            img = cv2.imread(image)
            facesCurFrame = fr.face_locations(img)
            Known_encoding_values= fr.face_encodings(img, facesCurFrame)
        
        #print(file)
    matchinglist = fr.compare_faces(Known_encoding_values, encodings_tobe_Check)  
    if matchinglist == []:
        continue
    else:      
        i+=1
        print(matchinglist,i)
        for qwery in matchinglist:
            if qwery==True:
                true+=1
            else:
                false+=1

        print("acc =", (true/(true+false))*100,"%")