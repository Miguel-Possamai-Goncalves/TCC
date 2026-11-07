from pathlib import Path
from PIL import Image

DIRETORIO_PROJETO = Path(__file__).resolve().parent.parent

DIRETORIO_ORIGEM = (
    DIRETORIO_PROJETO
    / "DATASET"
    / "Dividido"
)

DIRETORIO_DESTINO = (
    DIRETORIO_PROJETO
    / "DATASET"
    / "Dividido_224"
)

TAMANHO = (224, 224)

QUALIDADE_JPEG = 95

EXTENSOES_IMAGEM = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

def listar_imagens(diretorio):
    return [
        caminho
        for caminho in diretorio.rglob("*")
        if caminho.is_file()
        and caminho.suffix.lower() in EXTENSOES_IMAGEM
    ]

def tamanho_diretorio(diretorio):
    total = 0
    if not diretorio.exists():
        return 0
    for arquivo in diretorio.rglob("*"):
        if arquivo.is_file():
            total += arquivo.stat().st_size
    return total

def formatar_bytes(valor):
    unidades = ["B", "KB", "MB", "GB", "TB"]
    tamanho = float(valor)
    for unidade in unidades:
        if tamanho < 1024:
            return f"{tamanho:.2f} {unidade}"
        tamanho /= 1024
    return f"{tamanho:.2f} PB"

print("=" * 70)
print("CRIAÇÃO DA CÓPIA REDUZIDA DO DATASET")
print("=" * 70)

if not DIRETORIO_ORIGEM.exists():
    raise FileNotFoundError(
        f"\nPasta de origem não encontrada:\n{DIRETORIO_ORIGEM}\n\n"
        "Verifique se o script está dentro da pasta 'scripts' "
        "do projeto e se DATASET/Dividido existe."
    )

if DIRETORIO_DESTINO.exists():
    resposta = input(
        "\nA pasta de destino já existe. "
        "Deseja continuar e sobrescrever os arquivos existentes? [s/N]: "
    ).strip().lower()
    if resposta not in {"s", "sim"}:
        print("\nOperação cancelada.")
        raise SystemExit

imagens = listar_imagens(DIRETORIO_ORIGEM)

if not imagens:
    raise RuntimeError(
        f"Nenhuma imagem foi encontrada em:\n{DIRETORIO_ORIGEM}"
    )

print(f"\nImagens encontradas: {len(imagens):,}")
print(f"Origem:  {DIRETORIO_ORIGEM}")
print(f"Destino: {DIRETORIO_DESTINO}")
print(f"Tamanho final: {TAMANHO[0]}x{TAMANHO[1]}")
print(f"Qualidade JPEG: {QUALIDADE_JPEG}")

tamanho_original = tamanho_diretorio(DIRETORIO_ORIGEM)
print(f"\nTamanho atual do conjunto: {formatar_bytes(tamanho_original)}")

processadas = 0
erros = 0

for arquivo_origem in imagens:
    try:
        caminho_relativo = arquivo_origem.relative_to(DIRETORIO_ORIGEM)
        arquivo_destino = (
            DIRETORIO_DESTINO
            / caminho_relativo.parent
            / f"{arquivo_origem.stem}.jpg"
        )

        arquivo_destino.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(arquivo_origem) as imagem:
            imagem = imagem.convert("RGB")
            imagem = imagem.resize(TAMANHO, Image.Resampling.LANCZOS)
            imagem.save(
                arquivo_destino,
                format="JPEG",
                quality=QUALIDADE_JPEG,
                optimize=True
            )

        processadas += 1

        if processadas == 1 or processadas % 100 == 0 or processadas == len(imagens):
            porcentagem = (processadas / len(imagens)) * 100
            print(
                f"\rProcessando: {processadas:,}/{len(imagens):,} "
                f"({porcentagem:.1f}%)",
                end="",
                flush=True
            )

    except Exception as erro:
        erros += 1
        print(f"\n\nERRO: {arquivo_origem}")
        print(f"Detalhes: {erro}")

print("\n")

tamanho_destino = tamanho_diretorio(DIRETORIO_DESTINO)

print("=" * 70)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 70)
print(f"\nImagens processadas: {processadas:,}")
print(f"Erros:               {erros:,}")
print(f"\nTamanho original: {formatar_bytes(tamanho_original)}")
print(f"Tamanho reduzido: {formatar_bytes(tamanho_destino)}")

if tamanho_original > 0:
    reducao = (1 - (tamanho_destino / tamanho_original)) * 100
    print(f"Redução aproximada: {reducao:.1f}%")

print(f"\nNova cópia criada em:\n{DIRETORIO_DESTINO}")
print("\nIMPORTANTE:")
print("- DATASET/Dividido original não foi alterado.")
print("- A estrutura de Treino, Validação e Teste foi preservada.")
print("- As imagens foram convertidas para RGB.")
print("- As imagens foram redimensionadas para 224x224.")
print("- Os arquivos da nova cópia foram salvos em JPEG.")
