# Módulo padrão do Python para ler argumentos passados pelo terminal
# Ex: python main.py foto.png → sys.argv[1] == "foto.png"
import sys

# Módulo padrão para trabalhar com caminhos de arquivo
from pathlib import Path

# É o OpenCV2 - Biblioteca mais famosa de visão computacional
import cv2

# Trabalha com arrays de números
import numpy as np

# Para exibir o comparativo visual lado a lado
import matplotlib.pyplot as plt

# Importa as funções dos nossos módulos
from detector import detect_skew_angle
from corrector import correct_skew


# =============================================================================
# RESPONSABILIDADE 1: LER ARGUMENTO DO TERMINAL
# =============================================================================

def parse_arguments() -> str:
    """
    Lê e valida o argumento passado pelo terminal.

    Returns:
        caminho da imagem passado pelo usuário
    """
    if len(sys.argv) < 2:
        print("Uso: python main.py <caminho_da_imagem>")
        print("Ex:  python main.py documento.png")
        sys.exit(1)

    return sys.argv[1]


# =============================================================================
# RESPONSABILIDADE 2: CARREGAR IMAGEM DO DISCO
# =============================================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Carrega uma imagem do disco.

    Args:
        image_path: caminho para o arquivo de imagem

    Returns:
        imagem como array numpy
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erro: não foi possível carregar a imagem '{image_path}'")
        print("Verifique se o caminho está correto e o formato é suportado.")
        sys.exit(1)

    return image


# =============================================================================
# RESPONSABILIDADE 3: SALVAR IMAGEM NO DISCO
# =============================================================================

def save_image(image: np.ndarray, original_path: str) -> str:
    """
    Salva a imagem corrigida com sufixo '_corrected' no mesmo diretório.

    Ex: documentos/foto.png → documentos/foto_corrected.png

    Args:
        image:         imagem corrigida como array numpy
        original_path: caminho da imagem original

    Returns:
        caminho onde a imagem foi salvo
    """
    path = Path(original_path)
    output_path = path.parent / f"{path.stem}_corrected{path.suffix}"
    cv2.imwrite(str(output_path), image)
    return str(output_path)


# =============================================================================
# RESPONSABILIDADE 4: EXIBIR COMPARATIVO VISUAL
# =============================================================================

def show_comparison(original: np.ndarray, corrected: np.ndarray, angle: float):
    """
    Exibe as imagens original e corrigida lado a lado.

    Args:
        original:  imagem antes da correção
        corrected: imagem após a correção
        angle:     ângulo detectado pelo detector
    """

    # OpenCV armazena imagens em BGR, mas matplotlib espera RGB
    # Sem essa conversão as cores ficam erradas (vermelho vira azul, etc.)
    def to_rgb(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Skew detectado: {angle:.2f}°", fontsize=14)

    axes[0].imshow(to_rgb(original))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(to_rgb(corrected))
    axes[1].set_title("Corrigida")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAESTRO: ORQUESTRA TUDO EM ORDEM
# =============================================================================

def main():
    """
    Orquestra o pipeline completo.
    Não tem lógica própria — só chama as funções na ordem certa.
    """
    image_path = parse_arguments()

    image = load_image(image_path)
    print(f"[1/3] Imagem carregada: {image_path}")

    angle = detect_skew_angle(image)
    print(f"[2/3] Ângulo de skew detectado: {angle:.2f}°")

    if abs(angle) < 0.5:
        print("      Skew insignificante (<0.5°), nenhuma correção necessária.")
        corrected = image
    else:
        corrected = correct_skew(image, angle)
        print(f"[3/3] Correção aplicada.")

    output_path = save_image(corrected, image_path)
    print(f"\n✔ Imagem salva em: {output_path}")

    show_comparison(image, corrected, angle)


if __name__ == "__main__":
    main()