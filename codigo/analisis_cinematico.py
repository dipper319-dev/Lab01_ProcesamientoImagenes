import json
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
        """Carga datos del JSON generado por deteccion.py"""
        try:
            with open(archivo_json, 'r') as f:
                self.datos = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: No se encontró {archivo_json}")
            print("   Primero debes ejecutar deteccion.py para generar el JSON")
            exit()

        self.fps        = self.datos['fps']
        self.centroides = np.array([(c['x'], c['y']) for c in self.datos['centroides']])
        self.tiempos    = np.array(self.datos['tiempos'])

        if len(self.centroides) == 0 or len(self.tiempos) == 0:
            print("❌ Error: no hay datos de centroides/tiempos para analizar")
            print("   Ejecuta primero deteccion.py y verifica que se detecte el vehículo")
            exit()

        if len(self.centroides) != len(self.tiempos):
            print("❌ Error: inconsistencia entre cantidad de centroides y tiempos")
            exit()

        # Resultados calculados (se llenan con los métodos)
        self.pixeles_por_metro  = None
        self.posicion_metros    = None
        self.posicion_suavizada = None
        self.velocidad          = None
        self.aceleracion        = None

        print(f"\n✓ Datos cargados correctamente:")
        print(f"  - FPS: {self.fps:.2f}")
        print(f"  - Centroides: {len(self.centroides)}")
        print(f"  - Duración: {self.tiempos[-1]:.2f}s")

    # ------------------------------------------------------------------
    def calibrar_escala(self, pixel_A, pixel_B, distancia_real_metros):
        """
        Calibra la escala de píxeles a metros.

        Parámetros:
            pixel_A               : coordenada X (px) del marcador A en el video
            pixel_B               : coordenada X (px) del marcador B en el video
            distancia_real_metros : distancia real medida en campo entre A y B (m)

        CORRECCIÓN: la distancia real ya no está hardcodeada; se pasa como
        parámetro para que el usuario pueda ingresar el valor correcto de su
        grabación específica.
        """
        if distancia_real_metros <= 0:
            print("❌ Error: la distancia real A-B debe ser mayor que cero")
            return None

        distancia_pixeles = abs(pixel_B - pixel_A)

        if distancia_pixeles == 0:
            print("❌ Error: Los puntos A y B no pueden estar en la misma posición")
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
        """Convierte posición de píxeles a metros (eje X, origen en primer frame)"""
        x_pixeles          = self.centroides[:, 0].astype(float)
        x_relativo         = x_pixeles - x_pixeles[0]
        self.posicion_metros = x_relativo * metros_por_pixel

    # ------------------------------------------------------------------
    def suavizar_señal(self, ventana=11, orden_polinomio=3):
        """
        Aplica filtro Savitzky-Golay a la posición antes de derivar.

        ¿Por qué es necesario?
        ----------------------
        El centroide calculado frame a frame tiene ruido de 1-2 píxeles por
        imprecisiones en la segmentación. Al derivar numéricamente, ese ruido
        se amplifica: una oscilación de 1px entre frames consecutivos a 30fps
        produce un pico de velocidad de ~0.03/dt m/s, y la segunda derivada
        (aceleración) queda completamente dominada por ruido. El filtro
        Savitzky-Golay ajusta un polinomio local y elimina ese ruido sin
        distorsionar la forma real de la curva.

        Parámetros:
            ventana          : número de frames usados en la ventana local
                               (debe ser impar, >= orden_polinomio + 2)
            orden_polinomio  : grado del polinomio de ajuste (3 es suficiente)
        """
        n = len(self.posicion_metros)

        # La ventana no puede ser mayor que la señal; ajustar si es necesario
        ventana = min(ventana, n if n % 2 != 0 else n - 1)
        if ventana < orden_polinomio + 2:
            # Si hay muy pocos datos, usar media móvil simple como fallback
            print("⚠  Pocos datos para Savitzky-Golay, usando media móvil simple")
            kernel = np.ones(5) / 5
            self.posicion_suavizada = np.convolve(
                self.posicion_metros, kernel, mode='same'
            )
        else:
            self.posicion_suavizada = savgol_filter(
                self.posicion_metros, ventana, orden_polinomio
            )

        print(f"\n✓ Señal suavizada (ventana={ventana}, orden={orden_polinomio})")

    # ------------------------------------------------------------------
    def calcular_velocidad(self):
        """
        Calcula velocidad instantánea por diferencias finitas hacia adelante
        sobre la señal SUAVIZADA (no sobre el ruido crudo).
        """
        if self.posicion_suavizada is None:
            print("⚠  Calculando velocidad sobre señal sin suavizar (no recomendado)")
            señal = self.posicion_metros
        else:
            señal = self.posicion_suavizada

        if len(señal) < 2 or len(self.tiempos) < 2:
            self.velocidad = np.zeros(len(señal))
            print("⚠  Datos insuficientes para derivada: velocidad asignada a cero")
            return

        desplazamiento = np.diff(señal)
        tiempo_diff    = np.diff(self.tiempos)

        velocidad       = np.zeros(len(señal))
        velocidad[:-1]  = desplazamiento / tiempo_diff
        velocidad[-1]   = velocidad[-2]   # Extrapolar último valor

        self.velocidad = velocidad

        print(f"\n=== VELOCIDAD ===")
        print(f"Velocidad promedio : {np.mean(self.velocidad):.4f} m/s")
        print(f"Velocidad máxima   : {np.max(self.velocidad):.4f} m/s")
        print(f"Velocidad mínima   : {np.min(self.velocidad):.4f} m/s")

    # ------------------------------------------------------------------
    def calcular_aceleracion(self):
        """
        Calcula aceleración instantánea como segunda derivada numérica
        sobre la velocidad ya calculada.
        """
        if self.velocidad is None:
            print("❌ Error: primero debes calcular la velocidad")
            return

        if len(self.velocidad) < 2 or len(self.tiempos) < 2:
            self.aceleracion = np.zeros(len(self.velocidad))
            print("⚠  Datos insuficientes para segunda derivada: aceleración asignada a cero")
            return

        velocidad_diff = np.diff(self.velocidad)
        tiempo_diff    = np.diff(self.tiempos)

        aceleracion      = np.zeros(len(self.velocidad))
        aceleracion[:-1] = velocidad_diff / tiempo_diff
        aceleracion[-1]  = aceleracion[-2]

        self.aceleracion = aceleracion

        print(f"\n=== ACELERACIÓN ===")
        print(f"Aceleración promedio : {np.mean(self.aceleracion):.4f} m/s²")
        print(f"Aceleración máxima   : {np.max(self.aceleracion):.4f} m/s²")
        print(f"Aceleración mínima   : {np.min(self.aceleracion):.4f} m/s²")

    # ------------------------------------------------------------------
    def identificar_tipo_movimiento(self):
        """
        Clasifica el movimiento usando umbrales RELATIVOS a la magnitud
        de la señal (en lugar de valores fijos arbitrarios).

        Criterio:
          - Si la desviación estándar de la aceleración es < 20% de la
            velocidad media → aceleración aproximadamente constante
          - Si además |a_promedio| < 5% de v_media → MRU
          - Si no → MRUA
          - Si la desviación es >= 20% → movimiento con a variable
        """
        if self.velocidad is None or self.aceleracion is None:
            print("❌ Error: debes calcular velocidad y aceleración antes de clasificar")
            return "No determinado", 0.0, 0.0

        v_media              = abs(np.mean(self.velocidad))
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
        return tipo, a_promedio, v_media

    # ------------------------------------------------------------------
    def modelo_teorico(self, tipo, v0, a0):
        """
        Genera la curva teórica de posición para superponer en la gráfica.

        Para MRU  : x(t) = v0 * t
        Para MRUA : x(t) = v0*t + 0.5*a0*t²
        """
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
        """
        Genera gráficas profesionales con:
          - Señal cruda (gris claro) vs señal suavizada (color)
          - Curva teórica superpuesta en la gráfica de posición
        """
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
        plt.savefig('graficas_cinematica.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráficas guardadas en: graficas_cinematica.png")
        plt.show()

    # ------------------------------------------------------------------
    def guardar_resultados_csv(self):
        """Guarda los resultados completos en CSV"""
        import csv

        with open('resultados_cinematicos.csv', 'w', newline='') as f:
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

        print("✓ Resultados guardados en: resultados_cinematicos.csv")


# ============================
# EJECUTAR ANÁLISIS
# ============================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("ANÁLISIS CINEMÁTICO — PUNTO 3")
    print("="*50)

    analisis = AnalisisCinematico("datos_cinematicos.json")

    # --- Calibración ---
    print("\n⚠️  CALIBRACIÓN REQUERIDA:")
    print("Observa en el video los marcadores A y B y anota sus coordenadas X en píxeles.")

    try:
        pixel_A = float(input("\nIngresa coordenada X del punto A (píxeles): "))
        pixel_B = float(input("Ingresa coordenada X del punto B (píxeles): "))
        dist_real = float(input("Ingresa la distancia real A-B (metros): "))
    except ValueError:
        print("❌ Error: debes ingresar números válidos")
        exit()

    metros_por_pixel = analisis.calibrar_escala(pixel_A, pixel_B, dist_real)
    if metros_por_pixel is None:
        exit()

    # --- Pipeline de cálculo ---
    analisis.calcular_posicion_metros(metros_por_pixel)
    analisis.suavizar_señal(ventana=11, orden_polinomio=3)   # ← suavizar ANTES de derivar
    analisis.calcular_velocidad()
    analisis.calcular_aceleracion()

    tipo_movimiento, a_promedio, v_media = analisis.identificar_tipo_movimiento()

    # --- Gráficas y exportación ---
    print("\n📊 Generando gráficas...")
    analisis.graficar_resultados(tipo_movimiento, a_promedio, v_media)
    analisis.guardar_resultados_csv()

    print("\n✅ ¡Análisis completado exitosamente!")
    print(f"\n📁 Archivos generados:")
    print(f"   - graficas_cinematica.png")
    print(f"   - resultados_cinematicos.csv")