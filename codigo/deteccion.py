import cv2
import numpy as np

# -----------------------------
# Variables globales
# -----------------------------

pausar_video = False
mostrar_pixel = False
x, y = 0, 0

# -----------------------------
# Función del mouse
# -----------------------------

def mouse_callback(event, _x, _y, flags, param):
    global pausar_video, mostrar_pixel, x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = _x, _y
        mostrar_pixel = True

    if event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video


# -----------------------------
# Abrir video
# -----------------------------

cap = cv2.VideoCapture("video/vehiculo_test.mp4")

if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

print("Video abierto correctamente")

# -----------------------------
# Crear ventanas
# -----------------------------

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 400, 300)

def nothing(x):
    pass

# -----------------------------
# Crear trackbars HSV
# -----------------------------

cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)

cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

# Kernel morfológico
kernel = np.ones((5,5), np.uint8)

# -----------------------------
# Loop principal
# -----------------------------

while True:

    if not pausar_video:
        ret, frame = cap.read()

        if not ret:
            break

        frame2 = np.copy(frame)

    # Mostrar pixel seleccionado
    if mostrar_pixel:

        pixel_color = frame[y, x]

        texto = f"Pos ({x},{y}) Color BGR: {pixel_color}"

        cv2.putText(frame2, texto, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    # Convertir a HSV
    hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # Leer valores de trackbars

    h_min = cv2.getTrackbarPos("H Min","Trackbars")
    h_max = cv2.getTrackbarPos("H Max","Trackbars")

    s_min = cv2.getTrackbarPos("S Min","Trackbars")
    s_max = cv2.getTrackbarPos("S Max","Trackbars")

    v_min = cv2.getTrackbarPos("V Min","Trackbars")
    v_max = cv2.getTrackbarPos("V Max","Trackbars")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])

    # Crear máscara

    mascara = cv2.inRange(hsv, lower, upper)

    # -----------------------------
    # Limpiar máscara (morfología)
    # -----------------------------

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    # -----------------------------
    # Detectar contornos
    # -----------------------------

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos) > 0:

        contorno = max(contornos, key=cv2.contourArea)

        if cv2.contourArea(contorno) > 500:

            M = cv2.moments(contorno)

            if M["m00"] != 0:

                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])

                # Dibujar centroide
                cv2.circle(frame2,(cx,cy),6,(0,0,255),-1)

                # Dibujar contorno
                cv2.drawContours(frame2,[contorno],-1,(0,255,0),2)

                cv2.putText(frame2,f"Centroide: ({cx},{cy})",
                            (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,255,0),2)

    # -----------------------------
    # Mostrar ventanas
    # -----------------------------

    cv2.imshow("Video", frame2)
    cv2.imshow("Mascara", mascara)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()