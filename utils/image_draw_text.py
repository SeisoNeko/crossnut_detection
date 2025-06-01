import cv2
from cv2.typing import MatLike
import math

### ref: https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
def draw_text(img, text,
          pos=(0, 0),
          FONT_SCALE = 6e-3,
          THICKNESS_SCALE = 2e-3,
          text_color=(0, 255, 0),
          bg_color=(0, 0, 0),
          alpha=0.7 ) -> MatLike:
     
    height, width = img.shape[:-1]
    x, y = pos

    font_scale = min(width, height) * FONT_SCALE
    font_thickness = math.ceil(min(width, height) * THICKNESS_SCALE)

    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
    text_w, text_h = text_size

    overlay = img.copy()
    output = img.copy()


    ##### corner
    cv2.rectangle(overlay, pos, (x + text_w + 6, y + text_h + 6), bg_color, -1)

    ##### putText
    cv2.putText(overlay, text, (x+3, y + text_h + math.ceil(font_scale) + 2), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, font_thickness)

    #### apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output