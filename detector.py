# É o OpenCV2 - Biblioteca mais famosa de visão computacional
# Tem funções prontas pra manipular imagens
import cv2
# Trabalha com arrays de números 
import numpy as np

def detect_skew_angle(image: np.ndarray) -> float:
    """
    Detecta o ângulo de inclinação (skew) de um documento.
    
    Args:
        image: imagem como array numpy (colorida ou cinza)
    
    Returns:
        ângulo de inclinação em graus
    """
    #Escala de cinza
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    #Detecção de borda (Canny)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    #Hough Transform
    lines = cv2.HoughLinesP (
        edges,
        rho=1,
        theta = np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        angles.append(angle)

        median_angle = np.median(angles)
        return median_angle