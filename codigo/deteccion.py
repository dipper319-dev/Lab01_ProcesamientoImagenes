import cv2
import numpy as np

#------------------------
#Para Los que usen esto por primera vez ejecuten esto en consola para bajar las librerias "pip install -r requirements.txt"
#------------------------



pausar_video = False
mostrar_pixel = False
x, y = 0, 0

# Listas para guardar datos cinematicos (usados en punto 3)
centroides = []   # Lista de (cx, cy) por frame
tiempos = []      # Tiempo en segundos de cada frame

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

# Leer FPS del video (necesario para convertir frames a segundos en punto 3)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30.0  # Valor por defecto si no se puede leer
    print(f"Advertencia: no se pudo leer FPS, usando {fps} por defecto")
else:
    print(f"Video abierto correctamente | FPS: {fps:.2f}")

frame_num = 0  # Contador de frames procesados

# -----------------------------
# Crear sustractor de fondo MOG2
# -----------------------------
# MOG2 aprende el fondo automaticamente en los primeros frames,
# lo que lo hace mucho más robusto que el umbralamiento HSV manual.
# history: cuántos frames usa para aprender el fondo
# varThreshold: sensibilidad para detectar cambios (menor = más sensible)
# detectShadows: True detecta sombras y las marca en gris (valor 127)

subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=50,
    detectShadows=True
)

# -----------------------------
# Crear ventanas
# -----------------------------

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 400, 400)

def nothing(x):
    pass

# -----------------------------
# Trackbars de ajuste fino HSV (opcionales, como complemento al MOG2)
# -----------------------------

# Umbral de area minima para considerar un contorno como vehiculo
cv2.createTrackbar("Area Min", "Trackbars", 500, 10000, nothing)

# Tamaño del kernel morfologico (impar: 1, 3, 5, 7...)
cv2.createTrackbar("Kernel", "Trackbars", 5, 15, nothing)

# Sensibilidad del sustractor de fondo (varThreshold)
cv2.createTrackbar("Sensibilidad", "Trackbars", 50, 200, nothing)

# Mezcla entre mascara MOG2 y filtro HSV (0 = solo MOG2, 100 = solo HSV)
cv2.createTrackbar("Mezcla HSV", "Trackbars", 0, 100, nothing)

# Trackbars HSV (activos solo cuando Mezcla HSV > 0)
cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

# -----------------------------
# Loop principal
# -----------------------------

frame2 = None  # Proteccion: evitar crash si el video inicia en pausa

while True:

    if not pausar_video:
        ret, frame = cap.read()

        if not ret:
            print("Fin del video.")
            break

        frame2 = np.copy(frame)

    # Proteccion: si aun no hay frame (video pausado desde el inicio), esperar
    if frame2 is None:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        continue

    # -----------------------------
    # Leer trackbars
    # -----------------------------

    area_min     = cv2.getTrackbarPos("Area Min", "Trackbars")
    kernel_size  = cv2.getTrackbarPos("Kernel", "Trackbars")
    sensibilidad = cv2.getTrackbarPos("Sensibilidad", "Trackbars")
    mezcla_hsv   = cv2.getTrackbarPos("Mezcla HSV", "Trackbars")

    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    # Actualizar sensibilidad del sustractor dinamicamente
    subtractor.setVarThreshold(max(sensibilidad, 1))

    # Kernel morfologico: debe ser impar y >= 1
    k = max(1, kernel_size)
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)

    # -----------------------------
    # Mostrar info del pixel seleccionado
    # -----------------------------

    if mostrar_pixel:
        pixel_color = frame2[y, x]
        # Tambien mostrar el valor HSV del pixel (util para ajustar trackbars)
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_BGR2HSV)[0][0]
        texto_bgr = f"BGR: {pixel_color} | HSV: {pixel_hsv}"
        cv2.putText(frame2, texto_bgr, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # -----------------------------
    # MASCARA 1: Background Subtraction (MOG2)
    # -----------------------------
    # MOG2 devuelve: 255 = primer plano, 127 = sombra, 0 = fondo
    # Eliminamos sombras convirtiendo 127 a 0

    mascara_mog2 = subtractor.apply(frame2)
    mascara_mog2 = cv2.threshold(mascara_mog2, 200, 255, cv2.THRESH_BINARY)[1]

    # -----------------------------
    # MASCARA 2: Filtro HSV (opcional, como refinamiento)
    # -----------------------------

    mascara_hsv = np.zeros_like(mascara_mog2)  # Vacia por defecto

    if mezcla_hsv > 0:
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mascara_hsv = cv2.inRange(hsv, lower, upper)

    # -----------------------------
    # Combinar mascaras segun el slider de mezcla
    # mezcla_hsv = 0:  solo MOG2
    # mezcla_hsv = 100: solo HSV
    # valores intermedios: AND de ambas (mas restrictivo)
    # -----------------------------

    if mezcla_hsv == 0:
        mascara = mascara_mog2
    elif mezcla_hsv == 100:
        mascara = mascara_hsv
    else:
        # AND: conserva solo lo que ambas mascaras detectan (reduce falsos positivos)
        mascara = cv2.bitwise_and(mascara_mog2, mascara_hsv)

    # -----------------------------
    # Operaciones morfologicas para limpiar la mascara
    # -----------------------------

    # Apertura: elimina ruido pequeno (puntos aislados)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    # Cierre: rellena huecos dentro del vehiculo
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    # Dilatacion extra para unir partes fragmentadas del vehiculo
    mascara = cv2.dilate(mascara, kernel, iterations=1)

    # -----------------------------
    # Detectar contornos
    # -----------------------------

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    centroide_detectado = False  # Flag para saber si hubo deteccion en este frame

    if len(contornos) > 0:

        contorno = max(contornos, key=cv2.contourArea)

        if cv2.contourArea(contorno) > area_min:

            M = cv2.moments(contorno)

            if M["m00"] != 0:

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Guardar centroide y tiempo (para uso en punto 3)
                if not pausar_video:
                    centroides.append((cx, cy))
                    tiempos.append(frame_num / fps)

                centroide_detectado = True

                # Dibujar bounding box del vehiculo
                x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(contorno)
                cv2.rectangle(frame2, (x_bb, y_bb), (x_bb + w_bb, y_bb + h_bb),
                              (255, 165, 0), 2)

                # Dibujar contorno del vehiculo
                cv2.drawContours(frame2, [contorno], -1, (0, 255, 0), 2)

                # Dibujar centroide
                cv2.circle(frame2, (cx, cy), 6, (0, 0, 255), -1)

                # Texto informativo en pantalla
                cv2.putText(frame2, f"Centroide: ({cx}, {cy})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                area_texto = f"Area: {int(cv2.contourArea(contorno))} px"
                cv2.putText(frame2, area_texto,
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar estado de deteccion
    estado = "Vehiculo detectado" if centroide_detectado else "Sin deteccion"
    color_estado = (0, 255, 0) if centroide_detectado else (0, 0, 255)
    cv2.putText(frame2, estado,
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

    # Mostrar numero de frame y tiempo actual
    tiempo_actual = frame_num / fps
    cv2.putText(frame2, f"Frame: {frame_num} | t: {tiempo_actual:.2f}s",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Modo de segmentacion activo
    modo = "MOG2" if mezcla_hsv == 0 else ("HSV" if mezcla_hsv == 100 else "MOG2+HSV")
    cv2.putText(frame2, f"Modo: {modo}",
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Incrementar contador de frames solo si el video no esta pausado
    if not pausar_video:
        frame_num += 1

    # -----------------------------
    # Mostrar ventanas
    # -----------------------------

    cv2.imshow("Video", frame2)
    cv2.imshow("Mascara", mascara)
    cv2.imshow("MOG2 Raw", mascara_mog2)  # Ver que detecta MOG2 sin procesar

    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC para salir
        break

    if key == ord('s'):  # 's' para guardar datos hasta el momento
        print(f"\n--- Datos guardados hasta frame {frame_num} ---")
        print(f"Total centroides registrados: {len(centroides)}")
        if centroides:
            print(f"Ultimo centroide: {centroides[-1]}")
            print(f"Tiempo total: {tiempos[-1]:.2f}s")

# -----------------------------
# Resultado final
# -----------------------------

print(f"\n=== Procesamiento finalizado ===")
print(f"FPS del video: {fps:.2f}")
print(f"Frames procesados: {frame_num}")
print(f"Centroides detectados: {len(centroides)}")

if len(centroides) > 0:
    print(f"Primer centroide: {centroides[0]} en t={tiempos[0]:.3f}s")
    print(f"Ultimo centroide: {centroides[-1]} en t={tiempos[-1]:.3f}s")
    print(f"\nEstos datos (centroides y tiempos) estan listos para el Punto 3.")

cap.release()
cv2.destroyAllWindows()