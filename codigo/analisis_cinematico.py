import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================
# PUNTO 3: ANÁLISIS CINEMÁTICO
# ============================
# Correcciones aplicadas:
#  1. Distancia A-B dinámica (ingresada por el usuario, no hardcodeada)
#  2. Suavizado Savitzky-Golay antes de derivar (evita señales "sierra eléctrica")
#  3. Umbrales de clasificación de movimiento relativos a la señal
#  4. Comparación con modelo cinemático teórico (MRUA/MRU)
#  5. Gráficas mejoradas con curva teórica superpuesta
# ============================


class AnalisisCinematico:
    def __init__(self, archivo_json):
        """Carga los datos del JSON y prepara variables para todo el análisis."""
        self.resultados_dir = "resultados"
        os.makedirs(self.resultados_dir, exist_ok=True)

        try:
            with open(archivo_json, 'r') as f:
                self.datos = json.load(f)
        except FileNotFoundError:
            print(f"Error: no se encontró {archivo_json}")
            print("   Primero debes ejecutar deteccion.py para generar el JSON")
            exit()

        self.fps        = self.datos['fps']
        self.centroides = np.array([(c['x'], c['y']) for c in self.datos['centroides']])
        self.tiempos    = np.array(self.datos['tiempos'])

        if len(self.centroides) == 0 or len(self.tiempos) == 0:
            print("Error: no hay datos de centroides/tiempos para analizar")
            print("   Ejecuta primero deteccion.py y verifica que se detecte el vehículo")
            exit()

        if len(self.centroides) != len(self.tiempos):
            print("Error: inconsistencia entre cantidad de centroides y tiempos")
            exit()

        # Resultados calculados (se llenan con los métodos)
        self.pixeles_por_metro  = None
        self.posicion_metros    = None
        self.posicion_suavizada = None
        self.velocidad          = None
        self.aceleracion        = None

        print(f"\nDatos cargados correctamente:")
        print(f"  - FPS: {self.fps:.2f}")
        print(f"  - Centroides: {len(self.centroides)}")
        print(f"  - Duración: {self.tiempos[-1]:.2f}s")

    # ------------------------------------------------------------------
    def depurar_trayectoria(self, ventana_estable=8):
        """Elimina detecciones espurias al inicio y outliers cinemáticos."""
        # 1) Busca el primer tramo temporal estable (sin saltos grandes y con
        #    dirección dominante consistente).
        # 2) Elimina picos aislados de desplazamiento entre frames.
        # La idea es limpiar la señal antes de derivar para que velocidad y
        # aceleración no salgan contaminadas por ruido.
        if len(self.centroides) < ventana_estable + 3:
            print("Advertencia: muy pocos puntos para depuración automática")
            return

        x = self.centroides[:, 0].astype(float)
        dx = np.diff(x)
        abs_dx = np.abs(dx)

        med = np.median(abs_dx)
        umbral_salto = max(8.0 * med, 25.0)

        dx_no_cero = dx[np.abs(dx) > 1e-6]
        direccion_ref = np.sign(np.median(dx_no_cero)) if dx_no_cero.size > 0 else 0.0

        inicio_estable = 0
        for i in range(0, len(dx) - ventana_estable + 1):
            tramo = dx[i:i + ventana_estable]
            mag_ok = np.all(np.abs(tramo) <= umbral_salto)

            if direccion_ref == 0.0:
                dir_ok = True
            else:
                signos = np.sign(tramo[np.abs(tramo) > 1e-6])
                dir_ok = True if signos.size == 0 else (np.mean(signos == direccion_ref) >= 0.70)

            if mag_ok and dir_ok:
                inicio_estable = i
                break

        removidos_inicio = int(inicio_estable)
        if inicio_estable > 0:
            self.centroides = self.centroides[inicio_estable:]
            self.tiempos = self.tiempos[inicio_estable:]

        if len(self.centroides) < 5:
            print("Advertencia: la depuración recortó demasiados puntos; se mantiene el tramo disponible")
            return

        x = self.centroides[:, 0].astype(float)
        dx = np.diff(x)
        abs_dx = np.abs(dx)
        med2 = np.median(abs_dx)
        umbral_outlier = max(10.0 * med2, 40.0)

        mask = np.ones(len(self.centroides), dtype=bool)
        idx_outliers = np.where(np.abs(dx) > umbral_outlier)[0] + 1
        mask[idx_outliers] = False

        removidos_outliers = int(np.sum(~mask))
        if len(self.centroides[mask]) >= 5:
            self.centroides = self.centroides[mask]
            self.tiempos = self.tiempos[mask]

        print("\n=== DEPURACIÓN DE TRAYECTORIA ===")
        print(f"Puntos removidos al inicio : {removidos_inicio}")
        print(f"Outliers removidos         : {removidos_outliers}")
        print(f"Puntos finales             : {len(self.centroides)}")

    # ------------------------------------------------------------------
    @staticmethod
    def _suavizado_ligero(signal, max_window=9, polyorder=2):
        """Aplica un suavizado leve para quitar ondulaciones pequeñas."""
        n = len(signal)
        if n < 5:
            return signal.copy()

        window = min(max_window, n if n % 2 != 0 else n - 1)
        if window < polyorder + 2:
            return signal.copy()

        return savgol_filter(signal, window, polyorder)

    # ------------------------------------------------------------------
    @staticmethod
    def _filtrar_outliers_mad(signal, factor=3.5):
        """Recorta picos extremos con MAD cuando hay saltos fuertes entre muestras."""
        mediana = np.median(signal)
        mad = np.median(np.abs(signal - mediana))
        if mad < 1e-9:
            return signal.copy()

        escala = 1.4826 * mad
        limite_inf = mediana - factor * escala
        limite_sup = mediana + factor * escala
        return np.clip(signal, limite_inf, limite_sup)

    # ------------------------------------------------------------------
    @staticmethod
    def _regularizar_monotonia(signal):
        """Mantiene una tendencia dominante para evitar zig-zag poco realista en posición."""
        if len(signal) < 3:
            return signal.copy()

        dif = np.diff(signal)
        dif_no_cero = dif[np.abs(dif) > 1e-9]
        if dif_no_cero.size == 0:
            return signal.copy()

        direccion = np.sign(np.median(dif_no_cero))
        if direccion >= 0:
            return np.maximum.accumulate(signal)
        return np.minimum.accumulate(signal)

    # ------------------------------------------------------------------
    def calibrar_escala(self, pixel_A, pixel_B, distancia_real_metros):
        """Calibra la escala de píxeles a metros usando los puntos A y B."""
        # Parámetros:
        # - pixel_A, pixel_B: coordenadas X (px) de los marcadores.
        # - distancia_real_metros: distancia real medida en campo entre A y B.
        # Esta calibración evita valores fijos y permite convertir px -> m.
        if distancia_real_metros <= 0:
            print("Error: la distancia real A-B debe ser mayor que cero")
            return None

        distancia_pixeles = abs(pixel_B - pixel_A)

        if distancia_pixeles == 0:
            print("Error: los puntos A y B no pueden estar en la misma posición")
            return None

        self.pixeles_por_metro = distancia_pixeles / distancia_real_metros
        metros_por_pixel       = 1.0 / self.pixeles_por_metro

        print(f"\n=== CALIBRACIÓN DE ESCALA ===")
        print(f"Distancia A-B (real)   : {distancia_real_metros} m")
        print(f"Distancia A-B (píxeles): {distancia_pixeles} px")
        print(f"Escala                 : {self.pixeles_por_metro:.2f} px/m")
        print(f"Escala                 : {metros_por_pixel:.6f} m/px")

        return metros_por_pixel

    # ------------------------------------------------------------------
    def calcular_posicion_metros(self, metros_por_pixel):
        """Convierte X de píxeles a metros usando el primer frame como referencia."""
        x_pixeles          = self.centroides[:, 0].astype(float)
        x_relativo         = x_pixeles - x_pixeles[0]
        self.posicion_metros = x_relativo * metros_por_pixel

    # ------------------------------------------------------------------
    def suavizar_señal(self, ventana=11, orden_polinomio=3):
        """Suaviza la posición con Savitzky-Golay antes de derivar."""
        # Primero se suaviza x(t) y después se deriva.
        # Si se deriva primero, el ruido de detección se amplifica bastante.
        # Parámetros:
        # - ventana: número de frames usados (impar).
        # - orden_polinomio: grado del polinomio local.
        n = len(self.posicion_metros)

        # La ventana no puede ser mayor que la señal; ajustar si es necesario
        ventana = min(ventana, n if n % 2 != 0 else n - 1)
        if ventana < orden_polinomio + 2:
            # Si hay muy pocos datos, usar media móvil simple como fallback
            print("Advertencia: pocos datos para Savitzky-Golay, usando media móvil simple")
            kernel_len = min(5, n)
            kernel = np.ones(kernel_len) / kernel_len
            self.posicion_suavizada = np.convolve(
                self.posicion_metros, kernel, mode='same'
            )
        else:
            self.posicion_suavizada = savgol_filter(
                self.posicion_metros, ventana, orden_polinomio
            )

        self.posicion_suavizada = self._regularizar_monotonia(self.posicion_suavizada)
        self.posicion_suavizada = self._suavizado_ligero(self.posicion_suavizada, max_window=15, polyorder=2)

        print(f"\nSeñal suavizada (ventana={ventana}, orden={orden_polinomio})")

    # ------------------------------------------------------------------
    def calcular_velocidad(self):
        """Calcula la velocidad instantánea con diferencias finitas hacia adelante."""
        # La velocidad queda con signo porque depende de la dirección del eje X.
        if self.posicion_suavizada is None:
            print("Advertencia: calculando velocidad sobre señal sin suavizar (no recomendado)")
            señal = self.posicion_metros
        else:
            señal = self.posicion_suavizada

        if len(señal) < 2 or len(self.tiempos) < 2:
            self.velocidad = np.zeros(len(señal))
            print("Advertencia: datos insuficientes para derivada; velocidad asignada a cero")
            return

        dt = np.diff(self.tiempos)
        if np.any(dt <= 0):
            print("Error: el vector de tiempos no es monótono creciente")
            self.velocidad = np.zeros(len(señal))
            return

        velocidad = np.zeros(len(señal))
        velocidad[:-1] = (señal[1:] - señal[:-1]) / dt
        velocidad[-1] = velocidad[-2]

        velocidad = self._filtrar_outliers_mad(velocidad, factor=3.0)
        self.velocidad = self._suavizado_ligero(velocidad, max_window=31, polyorder=2)

        print(f"\n=== VELOCIDAD ===")
        print(f"Velocidad promedio : {np.mean(self.velocidad):.4f} m/s")
        print(f"Velocidad máxima   : {np.max(self.velocidad):.4f} m/s")
        print(f"Velocidad mínima   : {np.min(self.velocidad):.4f} m/s")

    # ------------------------------------------------------------------
    def calcular_aceleracion(self):
        """Calcula la aceleración instantánea como segunda derivada numérica."""
        # Como es segunda derivada, esta señal es muy sensible al ruido.
        # Por eso antes se suaviza y se filtran outliers.
        if self.posicion_metros is None:
            print("Error: primero debes calcular la posición en metros")
            return

        señal = self.posicion_suavizada if self.posicion_suavizada is not None else self.posicion_metros

        if len(señal) < 3 or len(self.tiempos) < 3:
            self.aceleracion = np.zeros(len(señal))
            print("Advertencia: datos insuficientes para segunda derivada; aceleración asignada a cero")
            return

        dt = np.diff(self.tiempos)
        if np.any(dt <= 0):
            print("Error: el vector de tiempos no es monótono creciente")
            self.aceleracion = np.zeros(len(señal))
            return

        aceleracion = np.zeros(len(señal))
        dt0 = dt[:-1]
        dt1 = dt[1:]
        denom = dt0 * dt1
        aceleracion[:-2] = (señal[2:] - 2 * señal[1:-1] + señal[:-2]) / denom
        aceleracion[-2] = aceleracion[-3]
        aceleracion[-1] = aceleracion[-2]

        aceleracion = self._filtrar_outliers_mad(aceleracion, factor=3.0)
        self.aceleracion = self._suavizado_ligero(aceleracion, max_window=41, polyorder=2)

        print(f"\n=== ACELERACIÓN ===")
        print(f"Aceleración promedio : {np.mean(self.aceleracion):.4f} m/s²")
        print(f"Aceleración máxima   : {np.max(self.aceleracion):.4f} m/s²")
        print(f"Aceleración mínima   : {np.min(self.aceleracion):.4f} m/s²")

    # ------------------------------------------------------------------
    def identificar_tipo_movimiento(self):
        """Clasifica el movimiento con umbrales relativos de aceleración."""
        # Criterio:
        # - Si std(a) < 20% de v_media, se asume aceleración casi constante.
        # - Si además |a_promedio| < 5% de v_media, se clasifica como MRU.
        # - En caso contrario, se clasifica como MRUA.
        # - Si std(a) es alta, se reporta aceleración variable.
        if self.velocidad is None or self.aceleracion is None:
            print("Error: debes calcular velocidad y aceleración antes de clasificar")
            return "No determinado", 0.0, 0.0

        v_media_signed       = np.mean(self.velocidad)
        v_media              = abs(v_media_signed)
        a_promedio           = np.mean(self.aceleracion)
        desviacion_acel      = np.std(self.aceleracion)

        # Umbral relativo: 20% de la velocidad media (ajustable)
        umbral_variabilidad  = 0.20 * v_media if v_media > 0 else 0.1
        # Umbral para considerar aceleración "cero": 5% de v_media
        umbral_cero_acel     = 0.05 * v_media if v_media > 0 else 0.05

        print(f"\n=== TIPO DE MOVIMIENTO ===")
        print(f"Velocidad media          : {v_media:.4f} m/s")
        print(f"Aceleración promedio     : {a_promedio:.4f} m/s²")
        print(f"Desviación std acel      : {desviacion_acel:.4f} m/s²")
        print(f"Umbral variabilidad      : {umbral_variabilidad:.4f} m/s²")

        if desviacion_acel < umbral_variabilidad:
            if abs(a_promedio) < umbral_cero_acel:
                tipo = "Movimiento Rectilíneo Uniforme (MRU)"
            else:
                tipo = "Movimiento Rectilíneo Uniformemente Acelerado (MRUA)"
        else:
            tipo = "Movimiento con aceleración variable"

        print(f"Tipo detectado           : {tipo}")
        return tipo, a_promedio, v_media_signed

    # ------------------------------------------------------------------
    def modelo_teorico(self, tipo, v0, a0):
        """Genera la curva teórica de posición para comparar con la señal medida."""
        # Para MRU:  x(t) = v0 * t
        # Para MRUA: x(t) = v0*t + 0.5*a0*t²
        t = self.tiempos - self.tiempos[0]   # tiempo relativo al inicio

        if "No determinado" in tipo:
            pos_teorica = np.zeros_like(t)
            label_teorico = "Modelo teórico no determinado"
        elif "MRU" in tipo and "MRUA" not in tipo:
            pos_teorica = v0 * t
            label_teorico = f"MRU teórico  (v₀={v0:.2f} m/s)"
        else:
            pos_teorica = v0 * t + 0.5 * a0 * t**2
            label_teorico = f"MRUA teórico (v₀={v0:.2f} m/s, a={a0:.2f} m/s²)"

        return pos_teorica, label_teorico

    # ------------------------------------------------------------------
    def graficar_resultados(self, tipo_movimiento, a_promedio, v_media):
        """Grafica posición, velocidad y aceleración, con modelo teórico de referencia."""
        # En la primera gráfica se comparan señal cruda, señal suavizada y
        # trayectoria teórica para validar coherencia física del movimiento.
        pos_teorica, label_teorico = self.modelo_teorico(
            tipo_movimiento, v_media, a_promedio
        )

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('Análisis Cinemático del Vehículo — Punto 3',
                     fontsize=16, fontweight='bold')

        # ---- Gráfica 1: Posición ----
        axes[0].plot(self.tiempos, self.posicion_metros,
                     color='lightblue', linewidth=1, alpha=0.6, label='Posición cruda')
        axes[0].plot(self.tiempos, self.posicion_suavizada,
                     'b-', linewidth=2.5, label='Posición suavizada')
        axes[0].plot(self.tiempos, pos_teorica,
                     'k--', linewidth=2, label=label_teorico)
        axes[0].fill_between(self.tiempos, self.posicion_suavizada, alpha=0.15, color='blue')
        axes[0].set_ylabel('Posición (m)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(fontsize=10)
        axes[0].set_title('Posición del Centroide vs Tiempo', fontsize=12, fontweight='bold')

        # ---- Gráfica 2: Velocidad ----
        vel_promedio = np.mean(self.velocidad)
        axes[1].plot(self.tiempos, self.velocidad,
                     'g-', linewidth=2.5, label='Velocidad')
        axes[1].axhline(y=vel_promedio, color='r', linestyle='--', linewidth=2,
                        label=f'Promedio: {vel_promedio:.3f} m/s')
        axes[1].fill_between(self.tiempos, self.velocidad, alpha=0.15, color='green')
        axes[1].set_ylabel('Velocidad (m/s)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(fontsize=10)
        axes[1].set_title('Velocidad Instantánea vs Tiempo', fontsize=12, fontweight='bold')

        # ---- Gráfica 3: Aceleración ----
        acel_promedio = np.mean(self.aceleracion)
        axes[2].plot(self.tiempos, self.aceleracion,
                     'r-', linewidth=2.5, label='Aceleración')
        axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
        axes[2].axhline(y=acel_promedio, color='b', linestyle='--', linewidth=2,
                        label=f'Promedio: {acel_promedio:.3f} m/s²')
        axes[2].fill_between(self.tiempos, self.aceleracion, alpha=0.15, color='red')
        axes[2].set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Aceleración (m/s²)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].legend(fontsize=10)
        axes[2].set_title('Aceleración Instantánea vs Tiempo', fontsize=12, fontweight='bold')

        plt.tight_layout()
        ruta_grafica = os.path.join(self.resultados_dir, 'graficas_cinematica.png')
        plt.savefig(ruta_grafica, dpi=300, bbox_inches='tight')
        print(f"\nGráficas guardadas en: {ruta_grafica}")
        plt.show()

    # ------------------------------------------------------------------
    def guardar_resultados_csv(self):
        """Guarda en CSV todos los resultados numéricos para anexar al informe."""
        import csv

        ruta_csv = os.path.join(self.resultados_dir, 'resultados_cinematicos.csv')
        with open(ruta_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Tiempo (s)', 'Centroide X (px)', 'Centroide Y (px)',
                'Posición cruda (m)', 'Posición suavizada (m)',
                'Velocidad (m/s)', 'Aceleración (m/s²)'
            ])

            pos_suav = self.posicion_suavizada if self.posicion_suavizada is not None \
                       else self.posicion_metros

            for i in range(len(self.tiempos)):
                writer.writerow([
                    f"{self.tiempos[i]:.4f}",
                    f"{self.centroides[i, 0]:.1f}",
                    f"{self.centroides[i, 1]:.1f}",
                    f"{self.posicion_metros[i]:.4f}",
                    f"{pos_suav[i]:.4f}",
                    f"{self.velocidad[i]:.4f}",
                    f"{self.aceleracion[i]:.4f}"
                ])

            print(f"Resultados guardados en: {ruta_csv}")


# ============================
# EJECUTAR ANÁLISIS
# ============================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("ANÁLISIS CINEMÁTICO — PUNTO 3")
    print("="*50)

    ruta_json_entrada = os.path.join("resultados", "datos_cinematicos.json")
    analisis = AnalisisCinematico(ruta_json_entrada)

    # --- Calibración ---
    print("\nCalibración requerida:")
    print("Convención usada: A = cono derecho, B = cono izquierdo.")
    print("Anota las coordenadas X de A y B desde el video procesado.")

    try:
        pixel_A = float(input("\nIngresa coordenada X del punto A (cono derecho, píxeles): "))
        pixel_B = float(input("Ingresa coordenada X del punto B (cono izquierdo, píxeles): "))
        dist_real = float(input("Ingresa la distancia real A-B (metros): "))
    except ValueError:
        print("Error: debes ingresar números válidos")
        exit()

    metros_por_pixel = analisis.calibrar_escala(pixel_A, pixel_B, dist_real)
    if metros_por_pixel is None:
        exit()

    distancia_estimada_recorrido = (np.max(analisis.centroides[:, 0]) - np.min(analisis.centroides[:, 0])) * metros_por_pixel
    umbral_recorrido_m = max(20.0, 1.5 * dist_real)
    if distancia_estimada_recorrido > umbral_recorrido_m:
        print(f"Advertencia: el recorrido estimado del vehículo ({distancia_estimada_recorrido:.2f} m) supera {umbral_recorrido_m:.2f} m.")
        print("   Revisa Distancia A-B real ingresada y coordenadas A/B (posible descalibración).")

    # --- Pipeline de cálculo ---
    analisis.depurar_trayectoria(ventana_estable=8)
    analisis.calcular_posicion_metros(metros_por_pixel)
    analisis.suavizar_señal(ventana=21, orden_polinomio=3)   # ← suavizar ANTES de derivar
    analisis.calcular_velocidad()
    analisis.calcular_aceleracion()

    tipo_movimiento, a_promedio, v_media = analisis.identificar_tipo_movimiento()

    # --- Gráficas y exportación ---
    print("\nGenerando gráficas...")
    analisis.graficar_resultados(tipo_movimiento, a_promedio, v_media)
    analisis.guardar_resultados_csv()

    print("\nAnálisis completado exitosamente.")
    print(f"\nArchivos generados:")
    print(f"   - resultados/graficas_cinematica.png")
    print(f"   - resultados/resultados_cinematicos.csv")