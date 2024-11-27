import cv2
import cvzone
import os
from cvzone.ClassificationModule import Classifier

class WasteSorter:
    def __init__(self, model_path, label_path, waste_folder, bins_folder, arrow_path, background_path, class_dic):
        self.classifier = Classifier(model_path, label_path)
        self.arrow_img = cv2.imread(arrow_path, cv2.IMREAD_UNCHANGED)
        self.img_background_path = background_path
        self.cap = cv2.VideoCapture(0)
        self.class_dic = class_dic

        # Load waste images
        self.img_waste_list = self._load_images(waste_folder)
        # Load bin images
        self.img_bins_list = self._load_images(bins_folder)

    @staticmethod
    def _load_images(folder_path):
        img_list = []
        path_list = os.listdir(folder_path)
        for path in sorted(path_list):  # Ensure consistent ordering
            img = cv2.imread(os.path.join(folder_path, path), cv2.IMREAD_UNCHANGED)
            img_list.append(img)
        return img_list

    def process_frame(self):
        _, img = self.cap.read()
        img_resize = cv2.resize(img, (454, 340))
        img_background = cv2.imread(self.img_background_path)

        prediction = self.classifier.getPrediction(img)
        print(prediction)

        class_id = prediction[1]

        if class_id != 0:
            # Ensure the selected waste image has 4 channels before overlaying
            if self.img_waste_list[class_id - 1].shape[2] == 3:
                self.img_waste_list[class_id - 1] = cv2.cvtColor(self.img_waste_list[class_id - 1], cv2.COLOR_BGR2BGRA)

            img_background = cvzone.overlayPNG(img_background, self.img_waste_list[class_id - 1], (909, 127))
            img_background = cvzone.overlayPNG(img_background, self.arrow_img, (978, 320))

            class_id_bin = self.class_dic.get(class_id, 0)  # Default to bin 0 if not found
            img_background = cvzone.overlayPNG(img_background, self.img_bins_list[class_id_bin], (920, 374))

        img_background[148:148 + 340, 159:159 + 454] = img_resize
        return img_background

    def run(self):
        while True:
            img_background = self.process_frame()
            cv2.imshow("Output", img_background)

            # Check for exit conditions
            key = cv2.waitKey(1)
            if key == 27 or cv2.getWindowProperty("Output", cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()
