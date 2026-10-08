      
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用 vLLM 判断图片中是否有人（监听版）：

  1) 递归扫描 ROOT/INPUT 下全部图片（含子目录）
  2) 仅识别 has_person
  3) 相对路径镜像到 ROOT 下 JSON / OUTPUT；按有无人分桶
  4) 可复制、移动图片，或仅创建同名空 txt
  5) WATCH_MODE 持续轮询新图
  6) 监听模式下单轮/单张异常只记日志，不退出；下轮继续扫漏检
  7) OUTPUT_OPERATION=copy|move|empty_txt

单文件可独立外发，第三方依赖：opencv-python / openai。

最终输出结构（相对 ROOT）：

  {INPUT_REL}/                      # 输入，递归查找
    batch_a/foo.jpg
    batch_b/sub/bar.png

  {JSON_REL}/                       # 识别结果，镜像输入相对路径
    batch_a/foo.json
    batch_b/sub/bar.json

  {OUTPUT_REL}/
    has_person/batch_a/foo.jpg        # 有人，保留输入相对路径
    no_person/batch_b/sub/bar.png     # 无人，保留输入相对路径
    # empty_txt 模式下对应保存为 foo.txt / bar.txt
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, Optional

import cv2
from openai import OpenAI

# -------------------------- 全局配置（按需修改） --------------------------
# 数据根目录；下面三个为相对 ROOT 的路径
ROOT_DIR = "/home/huangwenhua/project/dataset/head/tmp/"
INPUT_REL = "v003_neg"
JSON_REL = "json"
# 按分类桶直接保存图片，保留输入相对路径和原文件名
OUTPUT_REL = "output"

INPUT_DIR = os.path.join(ROOT_DIR, INPUT_REL)
JSON_DIR = os.path.join(ROOT_DIR, JSON_REL)
OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_REL)

# vLLM OpenAI 兼容接口
BASE_URL = os.getenv("VLLM_BASE_URL") or "http://192.168.202.101:8000/v1"
API_KEY = os.getenv("VLLM_API_KEY") or "EMPTY"
MODEL_NAME = os.getenv("VLLM_MODEL_NAME") or "qwen3.8-27b"

MAX_TOKENS = 256
TEMPERATURE = 0.0
MAX_IMAGE_SIZE = 1024
JPEG_QUALITY = 95

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
# vLLM 识别并发
DETECT_WORKERS = 2
STREAM = False
SKIP_EXISTING = True
ENABLE_THINKING = False
# copy=复制图片；move=移动图片；empty_txt=仅创建同名空 txt
OUTPUT_OPERATION: Literal["copy", "move", "empty_txt"] = "empty_txt"

# ---------- 持续监听 / 轮询 ----------
# True=持续递归扫描；False=只跑一轮
WATCH_MODE = True
POLL_INTERVAL_SEC = 30
# 忽略临时后缀（小写）
IGNORE_SUFFIXES = (".tmp", ".part", ".download", ".crdownload", ".aria2")

PROMPT_VERSION = "has_person_only_1.0"

_print_lock = threading.Lock()
ProcessStatus = Literal["success", "skip", "fail"]


def log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def save_output_image(image_path: str, output_path: str) -> None:
    """按配置保存分类图片，或创建同名空 txt。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if OUTPUT_OPERATION == "empty_txt":
        if not os.path.exists(output_path):
            with open(output_path, "w", encoding="utf-8"):
                pass
        return
    if os.path.abspath(image_path) != os.path.abspath(output_path):
        if OUTPUT_OPERATION == "copy" and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return
        shutil.copy2(image_path, output_path)


def remove_source_if_move(image_path: str) -> None:
    """OUTPUT_OPERATION=move 时删除输入源文件（输出侧已有副本）。"""
    if OUTPUT_OPERATION != "move":
        return
    try:
        if os.path.isfile(image_path):
            os.remove(image_path)
            log(f"[move] 已移除输入源：{image_path}")
    except OSError as exc:
        log(f"[警告] 移除输入源失败 {image_path}：{exc}")


# 内嵌提示词，不依赖外部文件
PROMPT = """
判断图片中是否存在真实人物。

判断规则：
- true：图片中存在清晰可辨的真实人物，包括成人、老人、青少年、儿童或婴儿。
- false：图片中不存在真实人物。
- 玩偶、雕塑、人体模型、卡通人物，以及照片、屏幕或印刷品中的人物不算。
- 即使只露出脸、头部、手脚或部分身体，只要能明确判断为现场真实人物，也返回 true。
- 无法确定时返回 false。

只输出合法 JSON，不要输出解释、Markdown 或其他字段：
{"has_person": false}
""".strip()


def resolve_output_path(bucket: str, image_filename: str) -> str:
    """按分类桶镜像相对路径；empty_txt 时改为 .txt。"""
    relative_path = image_filename
    if OUTPUT_OPERATION == "empty_txt":
        relative_path = f"{os.path.splitext(image_filename)[0]}.txt"
    return os.path.join(OUTPUT_DIR, bucket, relative_path)


def output_is_complete(output_path: str) -> bool:
    if OUTPUT_OPERATION == "empty_txt":
        return os.path.isfile(output_path)
    return os.path.isfile(output_path) and os.path.getsize(output_path) > 0


def list_images(input_dir: str) -> list[str]:
    """递归列出 input_dir 下图片，返回相对路径（相对 input_dir）。"""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    results: list[str] = []
    for root, _dirs, files in os.walk(input_dir):
        for name in sorted(files):
            if name.startswith("."):
                continue
            lower = name.lower()
            if any(lower.endswith(suf) for suf in IGNORE_SUFFIXES):
                continue
            if not lower.endswith(IMAGE_EXTENSIONS):
                continue
            full = os.path.join(root, name)
            if not os.path.isfile(full):
                continue
            results.append(os.path.relpath(full, input_dir))
    return sorted(results)


def resize_image_if_needed(
    image,
    max_size: int,
) -> tuple[Any, int, int, bool, float]:
    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= max_size:
        return image, width, height, False, 1.0

    scale = max_size / longest
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, new_width, new_height, True, scale


def load_and_encode_image(image_path: str, max_size: int) -> Optional[dict]:
    image = cv2.imread(image_path)
    if image is None:
        return None

    original_height, original_width = image.shape[:2]
    processed, width, height, resized, scale = resize_image_if_needed(image, max_size)

    ok, encoded = cv2.imencode(
        ".jpg",
        processed,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        return None

    image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return {
        "image_url": f"data:image/jpeg;base64,{image_base64}",
        "original_width": original_width,
        "original_height": original_height,
        "sent_width": width,
        "sent_height": height,
        "resized": resized,
        "scale": scale,
    }


def _chat_extra_body() -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"enable_thinking": bool(ENABLE_THINKING)},
    }


def call_vllm(client: OpenAI, image_url: str, prompt: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    extra_body = _chat_extra_body()

    if STREAM:
        answer_parts: list[str] = []
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
            extra_body=extra_body,
        )
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                with _print_lock:
                    print(delta.content, end="", flush=True)
                answer_parts.append(delta.content)
        log("")
        return {"answer": "".join(answer_parts), "stream": True}

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        extra_body=extra_body,
    )
    message = response.choices[0].message
    answer = message.content or ""
    reasoning = getattr(message, "reasoning_content", None)
    if not answer and reasoning:
        answer = str(reasoning)
    result: dict[str, Any] = {
        "answer": answer,
        "stream": False,
    }
    if hasattr(response, "usage") and response.usage is not None:
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return result


def _iter_json_dicts(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    results: list[dict] = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, dict):
            results.append(obj)
        i = start + 1
    return results


def _normalize_bool(raw: Any) -> Optional[bool]:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if int(raw) == 0:
            return False
        if int(raw) == 1:
            return True
        return None
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("true", "1", "yes"):
            return True
        if text in ("false", "0", "no"):
            return False
    if isinstance(raw, list) and len(raw) == 1:
        return _normalize_bool(raw[0])
    return None


def parse_attrs_from_answer(answer: Any) -> Optional[dict[str, Any]]:
    """仅解析 has_person。"""
    candidates: list[dict] = []
    if isinstance(answer, dict):
        candidates.append(answer)
    elif answer:
        content = str(answer).strip()
        if "```" in content:
            for block in re.findall(
                r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE
            ):
                try:
                    obj = json.loads(block.strip())
                except (json.JSONDecodeError, TypeError):
                    obj = None
                if isinstance(obj, dict):
                    candidates.append(obj)
        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                candidates.append(obj)
        except (json.JSONDecodeError, TypeError):
            pass
        candidates.extend(_iter_json_dicts(content))

    best: Optional[dict] = None
    best_score = -1
    for obj in candidates:
        score = 1 if "has_person" in obj else 0
        if score > best_score:
            best = obj
            best_score = score
    if best is None or best_score <= 0:
        return None

    has_person = _normalize_bool(best.get("has_person"))
    if has_person is None:
        return None

    return {"has_person": has_person}


def save_result(output_dir: str, image_filename: str, result: dict) -> str:
    """image_filename 可为相对路径；JSON 镜像到 output_dir 下同相对路径。"""
    name_prefix = os.path.splitext(image_filename)[0]
    json_path = os.path.join(output_dir, f"{name_prefix}.json")
    os.makedirs(os.path.dirname(json_path) or output_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return json_path


def classify_image_work(rel: str) -> Literal["new", "done"]:
    """仅根据两个分类目录中是否已有对应输出判断是否完成。"""
    for bucket in ("has_person", "no_person"):
        if output_is_complete(resolve_output_path(bucket, rel)):
            return "done"
    return "new"


def refresh_completed_output(rel: str) -> bool:
    """仅在 move 模式覆盖已完成的分类图片，并删除重新出现的源图。"""
    if OUTPUT_OPERATION != "move":
        return True

    image_path = os.path.join(INPUT_DIR, rel)
    stem = os.path.splitext(rel)[0]
    json_path = os.path.join(JSON_DIR, f"{stem}.json")
    try:
        with open(json_path, encoding="utf-8") as f:
            prev = json.load(f)
        attrs = prev.get("attrs") or {}
        has_person = bool(attrs.get("has_person", prev.get("has_person")))
        bucket = "has_person" if has_person else "no_person"
        output_path = resolve_output_path(bucket, rel)

        save_output_image(image_path, output_path)
        remove_source_if_move(image_path)
        log(f"[更新图片] {rel} -> {output_path}")
        return True
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        log(f"[警告] 更新已处理原图失败 {rel}：{exc}")
        return False


def _run_detect_pipeline(
    image_files: list[str],
    *,
    detect_workers: int,
    stage_name: str,
) -> tuple[int, int, int]:
    """并发识别，返回 (success, skip, fail)。"""
    if not image_files:
        return 0, 0, 0

    success_count = 0
    skip_count = 0
    fail_count = 0
    total = len(image_files)
    log(f"[{stage_name}] {total} 张 | 识别并发={detect_workers}")

    with ThreadPoolExecutor(max_workers=detect_workers) as detect_pool:
        detect_futures = {
            detect_pool.submit(detect_single_image, name, idx, total): name
            for idx, name in enumerate(image_files, 1)
        }
        for fut in as_completed(detect_futures):
            image_filename = detect_futures[fut]
            try:
                status = fut.result()
            except Exception as exc:
                log(f"[错误] 识别任务异常 {image_filename}：{exc}")
                status = "fail"

            if status == "success":
                success_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                fail_count += 1

    return success_count, skip_count, fail_count


def detect_single_image(
    image_filename: str,
    idx: int,
    total: int,
) -> ProcessStatus:
    """调用 vLLM 判断 has_person，并按分类目录落盘。"""
    image_path = os.path.join(INPUT_DIR, image_filename)
    stem, ext = os.path.splitext(image_filename)
    if not ext:
        ext = ".jpg"
    ext = ext.lower()
    json_path = os.path.join(JSON_DIR, f"{stem}.json")

    log(f"[{idx}/{total}] [识别] 开始：{image_filename}")

    if SKIP_EXISTING and os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                prev = json.load(f)
            attrs_prev = prev.get("attrs") or {}
            has_person_prev = bool(attrs_prev.get("has_person", prev.get("has_person")))
            bucket = "has_person" if has_person_prev else "no_person"
            output_path = resolve_output_path(bucket, image_filename)
            done = output_is_complete(output_path)
            if done:
                log(f"[跳过] 结果已存在：{json_path}")
                remove_source_if_move(image_path)
                return "skip"
            save_output_image(image_path, output_path)
            remove_source_if_move(image_path)
            log(f"[补输出] 复用已有分类：{image_filename} -> {output_path}")
            return "skip"
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            pass

    image_meta = load_and_encode_image(image_path, MAX_IMAGE_SIZE)
    if image_meta is None:
        log(f"[错误] 图片读取或编码失败：{image_path}")
        return "fail"

    if image_meta["resized"]:
        log(
            f"[缩放] {image_filename}: "
            f"{image_meta['original_width']}x{image_meta['original_height']}"
            f" -> {image_meta['sent_width']}x{image_meta['sent_height']}"
        )

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        api_result = call_vllm(client, image_meta["image_url"], PROMPT)
    except Exception as exc:
        log(f"[错误] 调用 vLLM 失败 {image_filename}：{exc}")
        return "fail"

    answer = api_result.get("answer", "")
    attrs = parse_attrs_from_answer(answer)
    if attrs is None:
        log(f"[错误] 无法解析属性：{image_filename} | answer={answer[:200]!r}")
        result = {
            "image_path": image_path,
            "image_filename": image_filename,
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "answer": answer,
            "attrs": None,
            "has_person": None,
            "output_dir": None,
            "usage": api_result.get("usage"),
            "process_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        save_result(JSON_DIR, image_filename, result)
        return "fail"

    has_person = bool(attrs["has_person"])
    bucket = "has_person" if has_person else "no_person"
    output_path = resolve_output_path(bucket, image_filename)
    save_output_image(image_path, output_path)

    # 两类都直接保存原文件，不做裁剪和去水印。
    result = {
        "image_path": image_path,
        "image_filename": image_filename,
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "original_size": {
            "width": image_meta["original_width"],
            "height": image_meta["original_height"],
        },
        "sent_size": {
            "width": image_meta["sent_width"],
            "height": image_meta["sent_height"],
        },
        "resized": image_meta["resized"],
        "resize_scale": image_meta["scale"],
        "answer": answer,
        "attrs": attrs,
        "has_person": has_person,
        "output_dir": os.path.dirname(output_path),
        "output_operation": OUTPUT_OPERATION,
        "paths": {
            "image": output_path if OUTPUT_OPERATION != "empty_txt" else None,
            "empty_txt": output_path if OUTPUT_OPERATION == "empty_txt" else None,
        },
        "usage": api_result.get("usage"),
        "stream": api_result.get("stream", False),
        "process_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    save_result(JSON_DIR, image_filename, result)
    remove_source_if_move(image_path)
    label = "有人" if has_person else "无人"
    log(f"[{label}] {image_filename} -> {output_path}")
    return "success"


def _empty_round_stats(**extra: int) -> dict[str, int]:
    base = {
        "success": 0,
        "skip": 0,
        "fail": 0,
        "new_tasks": 0,
        "scanned": 0,
    }
    base.update(extra)
    return base


def process_images(*, quiet_banner: bool = False) -> dict[str, int]:
    """
    每轮顺序（新数据优先）：
      1) 无对应分类输出的图片 → 识别
      2) 补检：新数据阶段后仍无对应分类输出
    任一步异常只记日志并尽量继续，不向外抛。
    """
    try:
        if not os.path.isdir(INPUT_DIR):
            log(f"[警告] 输入目录不存在，本轮跳过：{INPUT_DIR}")
            return _empty_round_stats()

        try:
            all_images = list_images(INPUT_DIR)
        except Exception as exc:
            log(f"[警告] 扫描输入目录失败，本轮跳过：{exc}")
            return _empty_round_stats()

        if not SKIP_EXISTING:
            new_files = list(all_images)
            completed_files: list[str] = []
        else:
            new_files = []
            completed_files = []
            for rel in all_images:
                try:
                    work = classify_image_work(rel)
                    if work == "done":
                        completed_files.append(rel)
                    else:
                        new_files.append(rel)
                except Exception:
                    new_files.append(rel)

        detect_workers = max(1, DETECT_WORKERS)
        if not quiet_banner:
            log(f"vLLM 接口：{BASE_URL}")
            log(f"模型：{MODEL_NAME}")
            log(f"提示词版本：{PROMPT_VERSION}")
            log(f"enable_thinking：{ENABLE_THINKING}")
            log(f"max_tokens：{MAX_TOKENS}，最长边：{MAX_IMAGE_SIZE}")
            log(f"ROOT：{ROOT_DIR}")
            log(f"输入目录（递归）：{INPUT_REL} -> {INPUT_DIR}")
            log(f"JSON 输出（镜像相对路径）：{JSON_REL} -> {JSON_DIR}")
            log(f"输出操作：{OUTPUT_OPERATION}")
            log(
                f"分类输出：{OUTPUT_REL}/has_person|no_person/{{输入相对路径}} "
                f"-> {OUTPUT_DIR}"
            )

        success_count = 0
        skip_count = 0
        fail_count = 0

        # move：覆盖分类图片后删除重新出现的同路径源图；
        # copy：已有结果直接跳过，不覆盖输出，也不处理输入源。
        if OUTPUT_OPERATION == "move":
            for rel in completed_files:
                refresh_completed_output(rel)

        # ----- 阶段1：优先新数据 -----
        try:
            if new_files:
                s, k, f = _run_detect_pipeline(
                    new_files,
                    detect_workers=detect_workers,
                    stage_name="优先新数据",
                )
                success_count += s
                skip_count += k
                fail_count += f
            else:
                log("[优先新数据] 本轮无新图")
        except Exception as exc:
            log(f"[警告] 新数据处理阶段异常（继续补检）：{exc}")

        # ----- 阶段2：补检未完成的识别 -----
        try:
            # 新数据阶段之后再扫一遍，纳入本轮新失败的
            redetect_files = []
            for rel in all_images:
                try:
                    if classify_image_work(rel) != "done":
                        redetect_files.append(rel)
                except Exception:
                    redetect_files.append(rel)
            if redetect_files:
                s, k, f = _run_detect_pipeline(
                    redetect_files,
                    detect_workers=detect_workers,
                    stage_name="补检未完成",
                )
                success_count += s
                skip_count += k
                fail_count += f
            else:
                log("[补检未完成] 无需补检")
        except Exception as exc:
            log(f"[警告] 补检阶段异常：{exc}")

        log(
            f"本轮完成：成功 {success_count} | 跳过 {skip_count} | 失败 {fail_count}"
        )

        return {
            "success": success_count,
            "skip": skip_count,
            "fail": fail_count,
            "new_tasks": success_count + fail_count,
            "scanned": len(all_images),
        }
    except Exception as exc:
        log(f"[警告] 本轮处理出现未预期异常，已吞掉以便继续监听：{exc}")
        return _empty_round_stats()


def process_images_watch() -> None:
    """递归扫描；业务异常不退出。Ctrl+C / SIGTERM 立即结束进程。"""

    def _handle_signal(signum, _frame):
        # 立即硬退出，不等待本轮识别线程池收尾
        try:
            log(f"\n收到信号 {signum}，立即退出。")
        except Exception:
            pass
        os._exit(128 + int(signum))

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception as exc:
        log(f"[警告] 注册信号处理失败：{exc}")

    log("=" * 60)
    log("监听：每轮 优先新数据 → 补检未完成（异常不退出）")
    log("=" * 60)
    log(f"ROOT：{ROOT_DIR}")
    log(f"输入：{INPUT_DIR}")
    log(f"JSON：{JSON_DIR}")
    log(f"输出操作：{OUTPUT_OPERATION}")
    log(f"输出：{OUTPUT_DIR}/has_person|no_person/{{输入相对路径}}")
    log(f"轮询间隔：{POLL_INTERVAL_SEC}s")
    log("按 Ctrl+C 立即退出")
    log("=" * 60)

    round_idx = 0
    while True:
        round_idx += 1
        log(f"\n>>> 第 {round_idx} 轮 @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            process_images(quiet_banner=True)
        except Exception as e:
            log(f"[本轮异常] {e}（不退出，等待下一轮）")

        try:
            time.sleep(max(1.0, POLL_INTERVAL_SEC))
        except Exception as e:
            log(f"[警告] 轮询等待异常：{e}")


def main() -> None:
    try:
        if OUTPUT_OPERATION not in {"copy", "move", "empty_txt"}:
            raise ValueError(f"无效 OUTPUT_OPERATION：{OUTPUT_OPERATION}")
        if WATCH_MODE:
            process_images_watch()
        else:
            process_images(quiet_banner=False)
            log("\n处理完成。")
    except KeyboardInterrupt:
        log("\n收到 KeyboardInterrupt，立即退出。")
        os._exit(130)
    except Exception as exc:
        log(f"[致命] 未捕获异常：{exc}")
        if WATCH_MODE:
            log("5 秒后尝试重新进入监听…")
            try:
                time.sleep(5)
                process_images_watch()
            except Exception as exc2:
                log(f"[致命] 重启监听仍失败：{exc2}")


if __name__ == "__main__":
    main()

    