import re
import argparse
import gc
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from audio_separator.separator import Separator


# Папка, в которой лежит main.py
BASE_DIR = Path(__file__).resolve().parent

# Папки для автоматического скачивания и кэширования моделей
MODEL_DIRS = {
    "sw": BASE_DIR / "bs_roformer_sw",
    "vocal": BASE_DIR / "bs_roformer_vocal",
    "demucs": BASE_DIR / "htdemucs_6s",
}

def rename_stems(output_dir: Path):
    names = {
        "vocals": "vocal",
        "vocal": "vocal",
        "guitar": "guitar",
        "drums": "drum",
        "drum": "drum",
        "bass": "bass",
        "piano": "piano",
        "other": "other",
        "instrumental": "instrumental",
    }

    for file in output_dir.glob("*.wav"):
        tokens = re.findall(r"[a-z]+", file.stem.lower())

        stem = next(
            (names[token] for token in reversed(tokens) if token in names),
            None,
        )

        if stem:
            target = file.with_name(f"{stem}.wav")

            if target != file and not target.exists():
                file.rename(target)
def release_gpu():
    """Освобождает неиспользуемую память после модели."""
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_roformer_executable():
    """Находит команду bs-roformer-infer в текущем окружении."""
    executable = shutil.which("bs-roformer-infer")

    if executable:
        return executable

    filename = (
        "bs-roformer-infer.exe"
        if os.name == "nt"
        else "bs-roformer-infer"
    )

    candidate = Path(sys.executable).with_name(filename)

    if candidate.exists():
        return str(candidate)

    raise RuntimeError(
        "Не найдена команда bs-roformer-infer. "
        "Установи пакет в текущее окружение:\n"
        "python -m pip install bs-roformer-infer"
    )


def run_bs_roformer_sw(audio_path: Path, output_dir: Path):
    """
    BS-RoFormer-SW:
    вокал, ударные, бас, гитара, пианино и прочие инструменты.

    Веса автоматически скачиваются при первом запуске.
    """
    print("\n[1/3] Запуск BS-RoFormer-SW")

    output_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIRS["sw"].mkdir(parents=True, exist_ok=True)

    executable = get_roformer_executable()

    # Эта реализация принимает WAV-файлы из входной папки.
    # Создаём временную папку только с нужным аудио.
    with tempfile.TemporaryDirectory(
        prefix="roformer_input_"
    ) as temp:
        temp_dir = Path(temp)
        wav_path = temp_dir / "input.wav"

        print("Подготовка аудио для модели...")

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(audio_path),
                "-vn",
                "-ac", "2",
                "-ar", "44100",
                str(wav_path),
            ],
            check=True,
        )

        command = [
            executable,
            "--input_folder", str(temp_dir),
            "--store_dir", str(output_dir),
            "--models_dir", str(MODEL_DIRS["sw"]),
        ]

        subprocess.run(command, check=True)
        rename_stems(output_dir)
    print("BS-RoFormer-SW завершила работу.")
    release_gpu()


def run_audio_separator(
    audio_path: Path,
    model_filename: str,
    model_dir: Path,
    output_dir: Path,
    label: str,
):
    """Запускает модель из библиотеки audio-separator."""

    print(f"\nЗапуск {label}")

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        model_file_dir=str(model_dir),
        output_dir=str(output_dir),
        output_format="WAV",
    )

    try:
        # Если весов нет в кэше, библиотека скачает их сама.
        separator.load_model(model_filename=model_filename)

        # Обрабатываем только указанный файл.
        output_files = separator.separate(str(audio_path))
        rename_stems(output_dir)
        print(f"Результаты модели {label}:")

        for filename in output_files:
            print(f"  {filename}")

    finally:
        del separator
        release_gpu()



def main():
    # Папка, в которой находится main.py
    base_dir = Path(__file__).resolve().parent

    # Исходный аудиофайл в той же папке
    audio_path = base_dir / "song.mp3"

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Не найден файл: {audio_path}\n"
            "Положи song.mp3 рядом с main.py."
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg не найден. Установи его и добавь в PATH."
        )

    # Проверка GPU
    import torch

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA недоступна, обработка может идти на CPU.")

    # Каталоги моделей
    for directory in MODEL_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)

    # Результаты в папке рядом с song.mp3
    output_root = audio_path.parent / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Входной файл: {audio_path}")
    print(f"Результаты: {output_root}")

    # 1. BS-RoFormer-SW
    run_bs_roformer_sw(
        audio_path,
        output_root / "bs_roformer_sw",
    )

    # 2. BS-RoFormer Vocal
    run_audio_separator(
        audio_path=audio_path,
        model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        model_dir=MODEL_DIRS["vocal"],
        output_dir=output_root / "bs_roformer_vocal",
        label="BS-RoFormer Vocal",
    )

    # 3. HTDemucs 6s
    run_audio_separator(
        audio_path=audio_path,
        model_filename="htdemucs_6s.yaml",
        model_dir=MODEL_DIRS["demucs"],
        output_dir=output_root / "htdemucs_6s",
        label="HTDemucs 6s",
    )

    print(f"\nГотово! Результаты: {output_root}")


if __name__ == "__main__":
    main()
