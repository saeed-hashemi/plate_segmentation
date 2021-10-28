import numpy as np
import cv2
from utils.postprocess import sort_rect_points, calcPointsWH, is_plate_on_left

def put_logo(image, mask, logo):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rect = cv2.minAreaRect(cnts[0][0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    m_rect=sort_rect_points(box)
    
    width, height, angel=(round(rect[1][0]),round(rect[1][1]),rect[2]) if rect[1][0]>rect[1][1] else (round(rect[1][1]),round(rect[1][0]),rect[2])
    
    reverse_rotation = None
    if width/height<1.85:
      print("Detected Size ERROR!")
      return False, image
    else:
      # change direction of logo and if the plate has been rotated        
      pts = np.array(m_rect, np.int32)
      if calcPointsWH(pts):
          pass
      else:
          # rotate clockwise or counter-clockwise
          if is_plate_on_left(m_rect[1],mask.shape[1]):
            logo=cv2.rotate(logo, cv2.ROTATE_90_CLOCKWISE)
            reverse_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
          else:  
            logo=cv2.rotate(logo, cv2.ROTATE_90_COUNTERCLOCKWISE)
            reverse_rotation = cv2.ROTATE_90_CLOCKWISE

          # change width and height of min rect
          width, height = height, width

      
      if width<logo.shape[1]: # shrink
        logo = cv2.resize(logo, (width,height), interpolation = cv2.INTER_AREA)
      else:
        logo = cv2.resize(logo, (width,height), interpolation = cv2.INTER_LINEAR)
      
      logo_shape=logo.shape
      paper2=np.ones((mask.shape[0],mask.shape[1],3), dtype=np.uint8)
      
      
      paper2[:height,:width,:] = logo
      
      pts = np.float32([[0,0],[logo_shape[1],0],[logo_shape[1],logo_shape[0]],[0,logo_shape[0]]])

      M = cv2.getPerspectiveTransform(pts,m_rect)
      dst = cv2.warpPerspective(paper2,M,(paper2.shape[1],paper2.shape[0]))
      mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
      
      logo=cv2.bitwise_and(dst,mask)
    
      r1=cv2.bitwise_and(image,255-mask)
      r2=cv2.addWeighted(r1,1.,logo,1.,1)
      if reverse_rotation is not None:
          r2=cv2.rotate(r2, reverse_rotation)
      return True, r2