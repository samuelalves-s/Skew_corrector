# É o OpenCV2 - Biblioteca mais famosa de visão computacional
# Tem funções prontas pra manipular imagens
import cv2

# Trabalha com arrays de números
import numpy as np


def correct_skew(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Corrige o skew de uma imagem rotacionando ela pelo ângulo detectado.

    Args:
        image: imagem original como array numpy
        angle: ângulo retornado pelo detect_skew_angle()

    Returns:
        imagem corrigida como array numpy
    """

    # Dimensões da imagem
    # image.shape retorna (altura, largura, canais)
    # O [:2] pega só altura e largura, ignora canais
    (height, width) = image.shape[:2]

    # Centro da imagem — será o ponto de pivô da rotação
    # Rotacionar pelo centro evita que a imagem "saia" do frame
    center = (width // 2, height // 2)

    # getRotationMatrix2D monta a matriz de transformação afim 2x3:
    #
    #   [ cos θ  -sin θ  tx ]
    #   [ sin θ   cos θ  ty ]
    #
    # O -angle desfaz o skew detectado
    # scale=1.0 significa sem zoom
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale=1.0)

    # warpAffine aplica a matriz de rotação em cada pixel da imagem
    # INTER_CUBIC: interpolação cúbica — melhor qualidade que LINEAR ou NEAREST
    # BORDER_REPLICATE: as bordas que ficam "vazias" após a rotação
    #                   são preenchidas replicando o pixel da borda mais próxima
    corrected = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return corrected