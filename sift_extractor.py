
import cv2
from tqdm import tqdm

def extract_sift(images):
    sift = cv2.SIFT_create()
    return [sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)[1]
            for img in tqdm(images)]
