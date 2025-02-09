import cv2
import os
import imutils

def crop_img(img):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

    return new_img

def preprocess_images(training_dir, save_path, img_size=224):
    training_dir_list = os.listdir(training_dir)

    for dir_name in training_dir_list:
        dir_save_path = os.path.join(save_path, dir_name)
        dir_path = os.path.join(training_dir, dir_name)
        image_dir_list = os.listdir(dir_path)
        
        for img_name in image_dir_list:
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            new_img = crop_img(img)
            new_img = cv2.resize(new_img, (img_size, img_size))
            
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            cv2.imwrite(os.path.join(dir_save_path, img_name), new_img)

if __name__ == "__main__":
    training_dir = "/kaggle/input/brain-tumor-mri-images-44c"
    save_path = "/kaggle/working/cleaned"
    preprocess_images(training_dir, save_path)