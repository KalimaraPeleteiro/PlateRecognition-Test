# Versão 02

from ultralytics import YOLO

import cv2
import easyocr

reader = easyocr.Reader(["en"], gpu = False)
license_plate_model = YOLO("best.pt")
caption = cv2.VideoCapture("Teste1.mp4")

ret = True

results = list()            # Agora, armazenando todos os resultados.

while True:
    ret, frame = caption.read()
    if ret:
        detections = license_plate_model(frame)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if score > 0.9:
              license_plate = frame[int(y1):int(y2), int(x1):int(x2), :]
              cv2.imwrite("Teste 1 (Placa).png", license_plate)

              license_plate_modified = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
              _, license_plate_modified = cv2.threshold(license_plate_modified, 64, 255, cv2.THRESH_BINARY_INV)
              cv2.imwrite("Teste 1 (Placa Modificada).png", license_plate_modified)

              textResult = ""
              scoreResult = list()
              reads = reader.readtext(license_plate_modified)       # Não se faz mas uso de confiança em OCR
              for read in reads:
                  coordinates, text, score = read
                  print(text, score)
                  if len(text) > 1:
                    text = text.upper().replace(' ', '')
                    scoreResult.append(score)
                    textResult += text
                    print(f"Placa Detectada! -> {text}")
                    print(f"Score -> {score}")
              if textResult != "":
                results.append([textResult, scoreResult])
    else:               
        break           # Interrompendo após todos os frames

print(f"Busca finalizada. Resultado final:")

# Valores armazenados são ordenados, depois. Detecções de mais de um texto são somados e divididos.
dictionaryResult = dict()
for k, v in results:
  if len(v) > 1:
    dictionaryResult[k] = sum(v)/len(v)
  else:
    dictionaryResult[k] = v[0]


# Ao final, todos os resultados são entregues.
for k, v in sorted(dictionaryResult.items(), key=lambda kv: kv[1], reverse = True):
  print(f"{k} --> {v}")

