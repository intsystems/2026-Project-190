import os
import subprocess
import tarfile
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np


DEFAULT_IMAGE_NAME = "dibco:06-30"
CONTAINER_NAME = "dibco_demo"
TAR_PATH = Path("dibco_06-30.tar")
INPUT_IMAGE_PATH = Path("datasets/school_notebooks_RU/images_base/4_55.JPG")
WORK_DIR = Path("tmp_files/dibco_demo")
MOUNT_DIR_IN_CONTAINER = "/mnt/volume"
INPUT_NAME = "input.png"
OUTPUT_NAME = "output.png"
PREVIEW_PATH = Path("debug_images/dibco_preview.jpg")


def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Короткое описание:
        Выполняет shell-команду и возвращает результат.
    Вход:
        cmd (List[str]): команда и ее аргументы.
        check (bool): бросать исключение при ненулевом коде выхода.
    Выход:
        subprocess.CompletedProcess: объект с stdout и stderr.
    """
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ensure_docker_access() -> None:
    """
    Короткое описание:
        Проверяет доступ к Docker daemon.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    try:
        run_cmd(["docker", "info"], check=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").lower()
        if "permission denied" in stderr or "docker.sock" in stderr:
            docker_group = run_cmd(["getent", "group", "docker"], check=False)
            current_groups = run_cmd(["id", "-nG"], check=False)
            docker_group_text = docker_group.stdout.strip()
            current_groups_text = current_groups.stdout.strip()
            raise RuntimeError(
                "Нет доступа к Docker daemon (permission denied).\n"
                f"docker group: {docker_group_text}\n"
                f"current groups: {current_groups_text}\n"
                "Если usermod уже выполнен, обнови группы в текущем терминале:\n"
                "newgrp docker\n"
                "или запусти одной командой:\n"
                "sg docker -c '/home/sasha/Documents/venv/bin/python test_1.py'"
            ) from exc
        raise RuntimeError(f"Docker недоступен:\n{exc.stderr}") from exc


def read_repo_tags_from_tar(tar_path: Path) -> List[str]:
    """
    Короткое описание:
        Читает список docker-тегов из manifest.json внутри tar-архива образа.
    Вход:
        tar_path (Path): путь к tar-архиву.
    Выход:
        List[str]: список тегов, например ['dibco:06-30_2'].
    """
    tags: List[str] = []
    try:
        with tarfile.open(tar_path, "r") as tf:
            manifest_member = tf.extractfile("manifest.json")
            if manifest_member is None:
                return tags
            manifest = json.load(manifest_member)
    except (tarfile.TarError, OSError, json.JSONDecodeError):
        return tags

    if not isinstance(manifest, list):
        return tags
    for item in manifest:
        repo_tags = item.get("RepoTags", []) if isinstance(item, dict) else []
        for tag in repo_tags:
            if isinstance(tag, str):
                tags.append(tag)
    return tags


def ensure_image_loaded() -> str:
    """
    Короткое описание:
        Проверяет наличие docker-образа и при необходимости грузит его из tar.
        Возвращает фактический тег образа для запуска.
    Вход:
        отсутствует.
    Выход:
        str: тег docker-образа.
    """
    env_image = os.getenv("DIBCO_IMAGE_NAME", "").strip()
    tar_tags = read_repo_tags_from_tar(TAR_PATH) if TAR_PATH.exists() else []
    candidates: List[str] = []
    if env_image:
        candidates.append(env_image)
    candidates.append(DEFAULT_IMAGE_NAME)
    candidates.extend(tar_tags)

    # Убираем дубли, сохраняя порядок.
    unique_candidates: List[str] = []
    for tag in candidates:
        if tag and tag not in unique_candidates:
            unique_candidates.append(tag)

    for image_name in unique_candidates:
        inspect = run_cmd(["docker", "image", "inspect", image_name], check=False)
        if inspect.returncode == 0:
            return image_name

    if not TAR_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {TAR_PATH}. Сначала скачай/положи tar в корень проекта."
        )
    run_cmd(["docker", "load", "-i", str(TAR_PATH)], check=True)

    # После загрузки ищем рабочий тег снова.
    for image_name in unique_candidates:
        inspect = run_cmd(["docker", "image", "inspect", image_name], check=False)
        if inspect.returncode == 0:
            return image_name

    # На случай, если manifest не прочитался заранее, пробуем перечитать после load.
    tar_tags_after = read_repo_tags_from_tar(TAR_PATH)
    for image_name in tar_tags_after:
        inspect = run_cmd(["docker", "image", "inspect", image_name], check=False)
        if inspect.returncode == 0:
            return image_name

    raise RuntimeError(
        "Образ загружен, но не удалось определить его тег.\n"
        "Проверь вручную: docker images | grep dibco\n"
        "Или задай DIBCO_IMAGE_NAME=<repo:tag>."
    )


def ensure_container_running(host_mount_dir: Path, image_name: str) -> None:
    """
    Короткое описание:
        Создает контейнер (если нужно) и гарантирует, что он запущен.
    Вход:
        host_mount_dir (Path): путь на хосте, который монтируется в контейнер.
        image_name (str): тег docker-образа.
    Выход:
        отсутствует.
    """
    inspect = run_cmd(["docker", "inspect", CONTAINER_NAME], check=False)
    if inspect.returncode == 0:
        # Контейнер уже существует: проверим совместимость image/mount.
        info = json.loads(inspect.stdout)[0]
        current_image = str(info.get("Config", {}).get("Image", ""))
        mounts = info.get("Mounts", [])
        has_required_mount = False
        for mount in mounts:
            src = str(mount.get("Source", ""))
            dst = str(mount.get("Destination", ""))
            if src == str(host_mount_dir.resolve()) and dst == MOUNT_DIR_IN_CONTAINER:
                has_required_mount = True
                break

        if current_image != image_name or not has_required_mount:
            run_cmd(["docker", "rm", "-f", CONTAINER_NAME], check=True)
        else:
            # Контейнер совместим: просто запускаем, если остановлен.
            running_state = run_cmd(
                ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
                check=True,
            ).stdout.strip().lower()
            if running_state != "true":
                run_cmd(["docker", "start", CONTAINER_NAME], check=True)
            return

    # Контейнера нет (или пересоздаем), создаем новый.
    create = run_cmd(
        [
            "docker",
            "run",
            "-dt",
            "--name",
            CONTAINER_NAME,
            "-v",
            f"{host_mount_dir.resolve()}:{MOUNT_DIR_IN_CONTAINER}",
            image_name,
            "/bin/bash",
        ],
        check=False,
    )
    if create.returncode == 0:
        return

    stderr = create.stderr or ""
    # Частый кейс: конфликт имени из-за race или старого состояния docker.
    if "is already in use" in stderr.lower():
        running_state = run_cmd(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            check=True,
        ).stdout.strip().lower()
        if running_state != "true":
            run_cmd(["docker", "start", CONTAINER_NAME], check=True)
        return

    raise RuntimeError(
        "Не удалось создать/запустить контейнер dibco_demo.\n"
        f"docker run stderr:\n{stderr}"
    )


def run_dibco_in_container() -> None:
    """
    Короткое описание:
        Пытается запустить бинаризацию внутри контейнера несколькими типовыми командами.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    script = rf"""
set -e
IN="{MOUNT_DIR_IN_CONTAINER}/{INPUT_NAME}"
OUT="{MOUNT_DIR_IN_CONTAINER}/{OUTPUT_NAME}"
rm -f "$OUT"

try_cmd() {{
  "$@" >/tmp/dibco_run.log 2>&1 || return 1
  [ -f "$OUT" ] || return 1
  return 0
}}

# Официальный запуск из README SmartEngines/unetbin:
# docker exec dibco /root/environment/bin/python /root/evaluate_answer.py -i <input> -o <output>
if [ -x /root/environment/bin/python ] && [ -f /root/evaluate_answer.py ]; then
  try_cmd /root/environment/bin/python /root/evaluate_answer.py -i "$IN" -o "$OUT" && exit 0
  # fallback: иногда скрипт ожидает имена файлов из /mnt/volume
  (cd "{MOUNT_DIR_IN_CONTAINER}" && try_cmd /root/environment/bin/python /root/evaluate_answer.py -i "{INPUT_NAME}" -o "{OUTPUT_NAME}") && exit 0
fi

exit 2
"""

    result = run_cmd(
        ["docker", "exec", CONTAINER_NAME, "bash", "-lc", script],
        check=False,
    )
    if result.returncode != 0:
        diagnostics = run_cmd(
            ["docker", "exec", CONTAINER_NAME, "bash", "-lc", "ls -la /root /root/environment/bin | sed -n '1,160p'"],
            check=False,
        )
        raise RuntimeError(
            "Не удалось выполнить бинаризацию внутри dibco-контейнера.\n"
            "Проверь файлы/команды внутри контейнера:\n"
            f"docker exec -it {CONTAINER_NAME} /bin/bash\n\n"
            "Диагностика:\n"
            f"{diagnostics.stdout or diagnostics.stderr}"
        )


def main() -> None:
    """
    Короткое описание:
        Демонстрирует применение DIBCO docker-модели бинаризации.
    Вход:
        отсутствует.
    Выход:
        отсутствует.
    """
    ensure_docker_access()
    image_name = ensure_image_loaded()

    if not INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Не найдено входное изображение: {INPUT_IMAGE_PATH}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    input_image = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_COLOR)
    if input_image is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {INPUT_IMAGE_PATH}")

    input_save_path = WORK_DIR / INPUT_NAME
    cv2.imwrite(str(input_save_path), input_image)

    ensure_container_running(WORK_DIR, image_name=image_name)
    run_dibco_in_container()

    output_path = WORK_DIR / OUTPUT_NAME
    if not output_path.exists():
        raise RuntimeError(f"Контейнер отработал, но файл не создан: {output_path}")

    output_img = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    if output_img is None:
        raise RuntimeError(f"Не удалось прочитать результат бинаризации: {output_path}")

    # Небольшой превью-коллаж "до/после".
    preview_left = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    preview = np.hstack([preview_left, output_img])
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PREVIEW_PATH), preview)

    print(f"[OK] Бинаризация готова: {output_path}")
    print(f"[OK] Превью сохранено: {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
