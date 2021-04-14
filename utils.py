import cv2
from imutils import face_utils
from math import asin, sin, degrees, radians, cos
import numpy as np 

def draw_points(image, shape):

    shape = face_utils.shape_to_np(shape)
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    return image

def get_yaw(shape):

    return degrees(asin((((shape.part(30).x - shape.part(2).x) / (shape.part(14).x - shape.part(2).x)) - 0.5) * 2 * sin(radians(60))))

def get_pitch(shape):
    return degrees(asin((((shape.part(33).y - shape.part(24).y) / (shape.part(8).y - shape.part(24).y)) - 0.5) * 2 * sin(radians(60))))

class Prop(object):


    def __init__(self, image_path, prop_type):

        self.prop_image_path = image_path
        
        self.point_param_dict = {"mustache": {"centerx_point":51, "centery_point":33, "centex1_point":31, "centex2_point":35, "width_multiple":3, "height_add":0.3, "width_add":0},
                                 "glasses": {"centerx_point":27, "centery_point":39, "centex1_point":16, "centex2_point":1, "width_multiple":1, "height_add":0, "width_add":0},
                                 "hat": {"centerx_point":27, "centery_point":27, "centex1_point":0, "centex2_point":16, "width_multiple":1.2, "height_add":-0.5, "width_add":0},
                                 "face_mask": {"centerx_point":None, "centery_point":None, "centex1_point":None, "centex2_point":None, "width_multiple":None, "height_add":None, "width_add":None}}
        
        self.set_point_params(prop_type)
        self.load_prop(self.prop_image_path)


    def set_point_params(self, prop_type):

        self.centerx_point = self.point_param_dict[prop_type]["centerx_point"]
        self.centery_point = self.point_param_dict[prop_type]["centery_point"]
        self.centex1_point = self.point_param_dict[prop_type]["centex1_point"]
        self.centex2_point = self.point_param_dict[prop_type]["centex2_point"]
        self.width_multiple = self.point_param_dict[prop_type]["width_multiple"]
        self.height_add = self.point_param_dict[prop_type]["height_add"]
        self.width_add = self.point_param_dict[prop_type]["width_add"]


    def load_prop(self, image_path):

        self.prop_image_fourchan = cv2.imread(image_path, -1)
        self.orig_mask = self.prop_image_fourchan[:,:,3]
        self.orig_mask_inv = cv2.bitwise_not(self.orig_mask)
        self.prop_image = self.prop_image_fourchan[:,:,0:3]
        self.orig_prop_height, self.orig_prop_width = self.prop_image.shape[:2]

        return None


    def apply_prop(self, image, shape):
        
        # calculate non-transformed prop dims
        pitch = get_pitch(shape)
        yaw = get_yaw(shape)

        self.prop_width = int(abs(self.width_multiple * (shape.part(self.centex1_point).x - shape.part(self.centex2_point).x)))
        self.prop_height = int(self.prop_width * self.orig_prop_height / self.orig_prop_width) - 10
        self.prop_height = int(self.prop_height * (2 - cos(radians(yaw))))
        x_shift = self.prop_width - int(self.prop_width * (sin(radians(yaw)) + 1))
        self.prop_width = int(self.prop_width * cos(radians(yaw)))

        prop = self.prop_image
        mask = self.orig_mask
        mask_inv = self.orig_mask_inv

        # pad image
        TRANSPARENT = [0, 0, 0]
        prop = cv2.copyMakeBorder(prop, int(prop.shape[0]/2), int(prop.shape[0]/2), 0, 0, cv2.BORDER_CONSTANT, value=TRANSPARENT)
        mask = cv2.copyMakeBorder(mask, int(mask.shape[0]/2), int(mask.shape[0]/2), 0, 0, cv2.BORDER_CONSTANT, value=0)
        mask_inv = cv2.copyMakeBorder(mask_inv, int(mask_inv.shape[0]/2), int(mask_inv.shape[0]/2), 0, 0, cv2.BORDER_CONSTANT, value=255)

        # perspective transform
        if yaw >= 0:
            pts1 = np.float32([[0,int(prop.shape[0]/4)],
                                [int(prop.shape[1]),int(prop.shape[0]/4) - (sin(radians(yaw)) * int(prop.shape[0]/4))],
                                [0,int(prop.shape[0]/4)*3],
                                [int(prop.shape[1]),int(prop.shape[0]/4)*3 + (sin(radians(yaw)) * int(prop.shape[0]/4))]])
        else:
            pts1 = np.float32([[0,int(prop.shape[0]/4) + (sin(radians(yaw)) * int(prop.shape[0]/4))],
                                [int(prop.shape[1]),int(prop.shape[0]/4)],
                                [0,int(prop.shape[0]/4)*3 - (sin(radians(yaw)) * int(prop.shape[0]/4))],
                                [int(prop.shape[1]),int(prop.shape[0]/4)*3]])

        pts2 = np.float32([[0,0],
                          [self.prop_width,0],
                          [0,self.prop_height],
                          [self.prop_width,self.prop_height]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        prop = cv2.warpPerspective(prop,M,(self.prop_width,self.prop_height))
        mask = cv2.warpPerspective(mask,M,(self.prop_width,self.prop_height))
        mask_inv = cv2.warpPerspective(mask_inv,M,(self.prop_width,self.prop_height))

        prop = cv2.resize(prop, (self.prop_width, self.prop_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.prop_width, self.prop_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(mask_inv, (self.prop_width, self.prop_height), interpolation = cv2.INTER_AREA)

        # calculate region of interest y1, y2, x1, x2
        y1 = int(shape.part(self.centery_point).y - (self.prop_height / 2) + (self.height_add * self.prop_height))
        y2 = int(y1 + self.prop_height)
        x1 = int(shape.part(self.centerx_point).x - (self.prop_width / 2) + (self.width_add * self.prop_width) + (x_shift/8))
        x2 = int(x1 + self.prop_width)

        # crop prop if it runs outside the image
        if y1 < 0:
            prop = prop[-y1:]
            mask = mask[-y1:]
            mask_inv = mask_inv[-y1:]
            y1 = 0

        if y2 > image.shape[0]:
            prop = prop[:image.shape[0]-y2]
            mask = mask[:image.shape[0]-y2]
            mask_inv = mask_inv[:image.shape[0]-y2]
            y2 = image.shape[0]

        if x1 < 0:
            prop = prop[:, -x1:]
            mask = mask[:, -x1:]
            mask_inv = mask_inv[:, -x1:]
            x1 = 0

        if x2 > image.shape[1]:
            prop = prop[:image.shape[1]-x2]
            mask = mask[:image.shape[1]-x2]
            mask_inv = mask_inv[:image.shape[1]-x2]
            x2 = image.shape[1]
        
        # apply prop onto image
        roi = image[max(0, y1):y2, max(0, x1):x2]
        roi_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
        roi_fg = cv2.bitwise_and(prop,prop,mask = mask)
        image[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

        return image


    
