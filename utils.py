import cv2

# Function to add text with background to the frame
def add_text_with_background(frame, text, font=cv2.FONT_HERSHEY_COMPLEX, font_scale=0.6, font_color=(255, 255, 255), font_thickness=1, bg_color=(0, 0, 0), bg_alpha=0.5, padding=10):

    # Get the width and height of the text box
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Calculate the position for the text (bottom-center)
    text_x = (frame.shape[1] - text_width) // 2
    text_y = frame.shape[0] - text_height - padding
    
    # Coordinates for the rectangle background
    rect_x1 = text_x - padding
    rect_y1 = text_y - text_height - padding // 2
    rect_x2 = text_x + text_width + padding
    rect_y2 = text_y + baseline + padding // 2
    
    # Create the rectangle background with alpha
    overlay = frame.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, cv2.FILLED)
    
    # Blend the rectangle with the frame using alpha
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    
    # Add the text on top of the rectangle
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return frame