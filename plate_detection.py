# Algoritmo
from ultralytics import YOLO

import cv2
import easyocr

reader = easyocr.Reader(["en"], gpu = False)
license_plate_model = YOLO("best.pt")
caption = cv2.VideoCapture("Teste01.mp4")

ret = True
located_plate = False

result = ""

while not located_plate:
    ret, frame = caption.read()
    if ret:
        detections = license_plate_model(frame)[0]          
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if score > 0.9:                                                 # Usando 90% de confiança.
              license_plate = frame[int(y1):int(y2), int(x1):int(x2), :]
              cv2.imwrite("Teste 01 (Placa).png", license_plate)

              license_plate_modified = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
              _, license_plate_modified = cv2.threshold(license_plate_modified, 64, 255, cv2.THRESH_BINARY_INV)
              cv2.imwrite("Teste 01 (Placa Modificada).png", license_plate_modified)

              reads = reader.readtext(license_plate_modified)
              for read in reads:
                  coordinates, text, score = read
                  if score > 0.8:                                           # Usando 80% de confiança.
                    text = text.upper().replace(' ', '')
                    result += text
                    print(f"Placa Detectada! -> {text}")
                    print(f"Score -> {score}")
                    located_plate = True

print(f"Busca finalizada. Resultado final: {result}") 
