"""
把影片靜音 → 依字幕檔燒上字幕 → 用 AI 在對應時間念出字幕。

用法：
    python make_video.py 1.mp4 1.txt
    python make_video.py 1.mp4 1.txt --voice zh-TW-YunJheNeural   (男聲)
    python make_video.py 1.mp4 1.srt                              (也可以用 .srt)

字幕 .txt 格式（一行一句：開始時間  結束時間  字幕文字）：
    0:01.5   0:04     大家好，歡迎來到我的頻道
    0:04.5   0:08.2   今天要介紹一個好用的工具
以 # 開頭的行會被忽略。時間可以寫 秒數(3.5)、分:秒(1:03.5) 或 時:分:秒。

AI 讀錯字時可以指定讀音：{字幕顯示的字|AI 念的字}
    點選右上{角|腳}的紀錄      → 字幕顯示「右上角」，AI 念「右上腳」

輸出：1_配音.mp4
"""
import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave

import edge_tts
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SR = 24000  # 配音取樣率
MAX_SPEEDUP = 1.6  # 配音太長時最多加速到幾倍
FONT_FILE = r"C:\Windows\Fonts\msjhbd.ttc"  # 微軟正黑體（粗）
FONT_NAME = "Microsoft JhengHei"


def parse_time(s):
    s = s.strip().replace(",", ".")
    parts = s.split(":")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return sec


def load_txt(path):
    items = []
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\S+)\s+(?:-+>?\s*)?(\S+)\s+(.+)$", line)
            if not m:
                sys.exit(f"第 {n} 行格式看不懂：{line}")
            try:
                start, end = parse_time(m.group(1)), parse_time(m.group(2))
            except ValueError:
                sys.exit(f"第 {n} 行時間寫錯：{line}")
            items.append((start, end, m.group(3).strip()))
    return items


def load_srt(path):
    with open(path, encoding="utf-8-sig") as f:
        blocks = re.split(r"\n\s*\n", f.read().strip())
    items = []
    for b in blocks:
        lines = b.strip().splitlines()
        for i, l in enumerate(lines):
            if "-->" in l:
                a, z = l.split("-->")
                text = " ".join(lines[i + 1:]).strip()
                if text:
                    items.append((parse_time(a), parse_time(z), text))
                break
    return items


READING = re.compile(r"\{([^{}|]*)\|([^{}]*)\}")


def display_text(t):
    return READING.sub(r"\1", t)


def speech_text(t):
    return READING.sub(r"\2", t)


def fmt_srt(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def video_duration(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True, errors="ignore")
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        sys.exit(f"讀不到影片長度：{path}")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def decode(path, tempo=1.0):
    """把音檔解碼成 numpy 陣列（單聲道 float32），可順便加速。"""
    cmd = [FFMPEG, "-v", "error", "-i", path]
    if tempo != 1.0:
        cmd += ["-af", f"atempo={tempo:.4f}"]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return trim_silence(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)


def trim_silence(a, thresh=0.01, keep=0.05):
    """剪掉語音前後的空白（各保留 0.05 秒）。"""
    loud = np.nonzero(np.abs(a) > thresh)[0]
    if len(loud) == 0:
        return a
    k = int(keep * SR)
    return a[max(0, loud[0] - k):min(len(a), loud[-1] + k)]


async def tts_all(items, voice, rate, outdir):
    sem = asyncio.Semaphore(4)

    async def one(i, text):
        path = os.path.join(outdir, f"line{i:03}.mp3")
        async with sem:
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(speech_text(text), voice, rate=rate).save(path)
                    return path
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"第 {i + 1} 句配音失敗：{e}")
                    await asyncio.sleep(2)

    return await asyncio.gather(*(one(i, t) for i, (_, _, t) in enumerate(items)))


def main():
    ap = argparse.ArgumentParser(description="影片靜音 + 字幕 + AI 配音")
    ap.add_argument("video")
    ap.add_argument("subs", help=".txt 或 .srt 字幕檔")
    ap.add_argument("--voice", default="zh-TW-HsiaoChenNeural",
                    help="配音聲音，例如 zh-TW-HsiaoChenNeural(女) zh-TW-HsiaoYuNeural(女) zh-TW-YunJheNeural(男)")
    ap.add_argument("--rate", default="+0%", help="語速，例如 +10%% 或 -10%%")
    ap.add_argument("--font-size", type=int, default=20, help="字幕大小")
    ap.add_argument("--margin", type=int, default=25, help="字幕離底部的距離，數字越小越靠下")
    ap.add_argument("--no-burn", action="store_true", help="不把字幕燒進畫面，改成可開關的字幕軌")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    items = load_srt(args.subs) if args.subs.lower().endswith(".srt") else load_txt(args.subs)
    if not items:
        sys.exit("字幕檔裡沒有任何句子")
    items.sort(key=lambda x: x[0])
    dur = video_duration(args.video)
    out = args.output or os.path.splitext(args.video)[0] + "_配音.mp4"

    for i, (s, e, t) in enumerate(items):
        if e <= s:
            sys.exit(f"第 {i + 1} 句結束時間要比開始時間晚：{t}")
        if s >= dur:
            sys.exit(f"第 {i + 1} 句開始時間 {s}s 超過影片長度 {dur:.1f}s：{t}")

    tmp = tempfile.mkdtemp(prefix="dub_")
    try:
        print(f"產生 {len(items)} 句 AI 配音（{args.voice}）...")
        mp3s = asyncio.run(tts_all(items, args.voice, args.rate, tmp))

        track = np.zeros(int((dur + 1) * SR), dtype=np.float32)
        for i, ((s, e, t), mp3) in enumerate(zip(items, mp3s)):
            clip = decode(mp3)
            # 可用時間：到這句結束，或到下一句開始（取較晚者，但不超過下一句開始）
            next_start = items[i + 1][0] if i + 1 < len(items) else dur
            slot = max(e, min(next_start, dur)) - s
            need = len(clip) / SR
            tempo = 1.0
            if need > slot:
                tempo = need / slot
                if tempo > MAX_SPEEDUP:
                    print(f"  ⚠ 第 {i + 1} 句太長（要 {need:.1f}s，只有 {slot:.1f}s），"
                          f"加速 {MAX_SPEEDUP} 倍後仍會蓋到下一句：{t}")
                    tempo = MAX_SPEEDUP
                clip = decode(mp3, tempo)
            a = int(s * SR)
            end = min(a + len(clip), len(track))
            track[a:end] += clip[:end - a]
            speed = f"（加速 {tempo:.2f} 倍）" if tempo > 1 else ""
            print(f"  {i + 1:>3}. {fmt_srt(s)[3:]}  {len(clip) / SR:4.1f}s  {display_text(t)}{speed}")

        track = np.clip(track, -1, 1)
        wav = os.path.join(tmp, "voice.wav")
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((track * 32767).astype(np.int16).tobytes())

        srt = os.path.join(tmp, "subs.srt")
        with open(srt, "w", encoding="utf-8") as f:
            for i, (s, e, t) in enumerate(items, 1):
                f.write(f"{i}\n{fmt_srt(s)} --> {fmt_srt(e)}\n{display_text(t)}\n\n")

        print("合成影片...")
        video_abs = os.path.abspath(args.video)
        out_abs = os.path.abspath(out)
        cmd = [FFMPEG, "-y", "-v", "error", "-stats", "-i", video_abs, "-i", "voice.wav"]
        if args.no_burn:
            cmd += ["-i", "subs.srt", "-map", "0:v", "-map", "1:a", "-map", "2:s",
                    "-c:v", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=chi"]
        else:
            shutil.copy(FONT_FILE, os.path.join(tmp, os.path.basename(FONT_FILE)))
            style = (f"FontName={FONT_NAME},FontSize={args.font_size},PrimaryColour=&H00FFFFFF,"
                     f"OutlineColour=&H00000000,BorderStyle=1,Outline=1.5,Shadow=0,MarginV={args.margin}")
            cmd += ["-map", "0:v", "-map", "1:a",
                    "-vf", f"subtitles=subs.srt:fontsdir=.:force_style='{style}'",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}", "-movflags", "+faststart", out_abs]
        # 在暫存資料夾裡執行，避免 Windows 路徑在字幕濾鏡裡出問題
        subprocess.run(cmd, cwd=tmp, check=True)
        print(f"\n完成：{out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
