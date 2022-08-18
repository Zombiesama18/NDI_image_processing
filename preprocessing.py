import cv2
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    figuresPath = Path(str(Path.joinpath(Path.cwd(), '20220725/20220725/Observed_Crop_200x200pix')))
    file_names = list(figuresPath.glob('*.jpg'))
    img_file = np.random.choice(file_names)
    img = cv2.imread(str(img_file))
    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()