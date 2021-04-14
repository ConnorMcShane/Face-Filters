import dlib
import cv2
from PIL import Image, ImageDraw
from utils import Prop, draw_points, get_pitch, get_yaw

gif_images = []

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

mustache = Prop("mustache.png", "mustache")
glasses = Prop("glasses.png", "glasses")
hat = Prop("hat.png", "hat")

cap = cv2.VideoCapture('selfie_video.MOV')

if (cap.isOpened()== False): 
  print("Error opening video  file")

img_counter = 0
frame_counter = 0

while(cap.isOpened()):

    ret, image = cap.read()

    if ret == True:
        
        # resize image
        shape = image.shape
        h = int(shape[0]/2)
        w = int(shape[1]/2)
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

        # set cap forward 3 frames and count number of imgs  processed
        img_counter += 1
        frame_counter += 3
        cap.set(1, frame_counter)

        # convert to grey and detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
            # predict face landmarks
            shape = predictor(gray, rect)
            
            image = mustache.apply_prop(image, shape)
            image = glasses.apply_prop(image, shape)
            image = hat.apply_prop(image, shape)

            # image = draw_points(image, shape)
            # yaw = get_yaw(shape)
            # pitch = get_pitch(shape)
            # cv2.putText(image, "Yaw: " + str(round(yaw, 2)) + " DEG", (10, int(image.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # cv2.putText(image, "Pitch: " + str(round(pitch, 2)) + " DEG", (10, int(image.shape[0]/2) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("img", image)
        cv2.waitKey(1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gif_images.append(Image.fromarray(image))
        
    else:
        cap.release()
        break

cv2.destroyAllWindows()
gif_images[0].save('out2.gif', save_all=True, append_images=gif_images[1:], optimize=False, duration=img_counter, loop=0)