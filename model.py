from ultralytics import YOLO
import pickle
model = YOLO("../models/l_version_1_300.pt")
pickle.dump(model, open("model.pkl","wb"))