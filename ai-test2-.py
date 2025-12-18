import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import simpleaudio as sa
import whisper
from pydub import AudioSegment
from TTS.api import TTS

# --- TTS ---
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True, gpu=False)
speakers = tts.speakers
default_speaker = speakers[0]


def browse_file():
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if path:
        entry_text.delete(0, tk.END)
        entry_text.insert(0, path)


def generate_voice():
    path_or_text = entry_text.get()
    if not path_or_text:
        messagebox.showerror("Ошибка", "Введите текст или выберите файл")
        return
    if os.path.isfile(path_or_text):
        with open(path_or_text, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = path_or_text
    speaker = speaker_combo.get()
    temp_wav = "temp.wav"

    tts.tts_to_file(text=text, speaker=speaker, file_path=temp_wav)

    save_path = filedialog.asksaveasfilename(
        defaultextension=".mp3",
        filetypes=[("MP3 files", "*.mp3"), ("WAV files", "*.wav")],
    )
    if save_path:
        if save_path.endswith(".mp3"):
            sound = AudioSegment.from_wav(temp_wav)
            sound.export(save_path, format="mp3")
            os.remove(temp_wav)
        else:
            os.rename(temp_wav, save_path)
        messagebox.showinfo("Готово", f"Аудио сохранено: {save_path}")
    else:
        os.remove(temp_wav)


def play_voice():
    if os.path.exists("temp.wav"):
        wave_obj = sa.WaveObject.from_wave_file("temp.wav")
        wave_obj.play()
    else:
        messagebox.showerror("Ошибка", "Сначала сгенерируйте голос")


# --- Видео из аудио с субтитрами через OpenCV ---
def open_video_creator():
    def create_video():
        audio_path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.mp3 *.wav")]
        )
        if not audio_path:
            return

        # Распознаем аудио
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        raw_segments = result.get("segments", [])
        if not raw_segments:
            messagebox.showerror("Ошибка", "Не удалось распознать речь.")
            return

        # Разбиваем длинные сегменты на более короткие (до 8 слов)
        segments = []
        for seg in raw_segments:
            words = seg["text"].split()
            start = seg["start"]
            end = seg["end"]
            duration = end - start
            chunk_count = max(1, len(words) // 8 + (1 if len(words) % 8 else 0))
            chunk_time = duration / chunk_count
            for i in range(chunk_count):
                part = " ".join(words[i * 8 : (i + 1) * 8])
                segments.append(
                    {
                        "text": part,
                        "start": start + i * chunk_time,
                        "end": start + (i + 1) * chunk_time,
                    }
                )

        duration = segments[-1]["end"]
        fps = 24
        frame_size = (1080, 1080)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_video_path = "temp_video.mp4"
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, frame_size)

        # Зеленый фон
        green_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        green_frame[:] = (0, 255, 0)

        # Настройки шрифта
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_font_scale = 1.8
        base_thickness = 4
        max_width = frame_size[0] - 100  # отступы по краям

        # Разбиение текста на строки по 5 слов
        def wrap_text(text, max_words=5):
            words = text.split()
            lines = []
            for i in range(0, len(words), max_words):
                lines.append(" ".join(words[i : i + max_words]))
            return lines

        # Рендеринг кадров
        total_frames = int(duration * fps)
        for i in range(total_frames):
            t = i / fps
            frame = green_frame.copy()

            for seg in segments:
                if seg["start"] <= t <= seg["end"]:
                    text = seg["text"].strip()
                    lines = wrap_text(text, max_words=5)

                    # Центровка блока текста
                    line_height = 80
                    total_text_height = len(lines) * line_height
                    start_y = (frame_size[1] - total_text_height) // 2

                    for j, line in enumerate(lines):
                        font_scale = base_font_scale
                        text_size = cv2.getTextSize(
                            line, font, font_scale, base_thickness
                        )[0]
                        while text_size[0] > max_width:
                            font_scale -= 0.1
                            text_size = cv2.getTextSize(
                                line, font, font_scale, base_thickness
                            )[0]
                            if font_scale < 0.8:
                                break

                        text_x = (frame_size[0] - text_size[0]) // 2
                        text_y = start_y + (j + 1) * line_height

                        # Тень / обводка
                        for dx in [-2, 2]:
                            for dy in [-2, 2]:
                                cv2.putText(
                                    frame,
                                    line,
                                    (text_x + dx, text_y + dy),
                                    font,
                                    font_scale,
                                    (0, 0, 0),
                                    base_thickness + 2,
                                    cv2.LINE_AA,
                                )
                        # Белый текст
                        cv2.putText(
                            frame,
                            line,
                            (text_x, text_y),
                            font,
                            font_scale,
                            (255, 255, 255),
                            base_thickness,
                            cv2.LINE_AA,
                        )
                    break

            video_writer.write(frame)

        video_writer.release()

        # Добавляем аудио через ffmpeg
        final_video_path = filedialog.asksaveasfilename(
            defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")]
        )
        if final_video_path:
            audio_out_path = "temp_audio.mp3"
            AudioSegment.from_file(audio_path).export(audio_out_path, format="mp3")
            os.system(
                f'ffmpeg -y -i {temp_video_path} -i {audio_out_path} -c:v copy -c:a aac -strict experimental "{final_video_path}"'
            )
            os.remove(temp_video_path)
            os.remove(audio_out_path)
            messagebox.showinfo("Готово", f"Видео сохранено: {final_video_path}")

    win = tk.Toplevel(root)
    win.title("Создать видео с субтитрами")
    tk.Label(win, text="Выберите аудио для создания видео").pack(pady=10)
    tk.Button(win, text="Выбрать аудио и создать видео", command=create_video).pack(
        pady=20
    )


# --- GUI ---
root = tk.Tk()
root.title("Text-to-Speech Generator")

tk.Label(root, text="Введите текст или выберите файл:").pack(pady=5)
entry_text = tk.Entry(root, width=60)
entry_text.pack(pady=5)

tk.Button(root, text="Выбрать файл", command=browse_file).pack(pady=5)

tk.Label(root, text="Выберите голос:").pack(pady=5)
speaker_combo = ttk.Combobox(root, values=speakers, state="readonly")
speaker_combo.set(default_speaker)
speaker_combo.pack(pady=5)

tk.Button(root, text="Сгенерировать голос", command=generate_voice).pack(pady=5)
tk.Button(root, text="Прослушать", command=play_voice).pack(pady=5)

tk.Label(root, text="Заметка").pack(pady=5)
notes_text = tk.Text(root, height=10, width=60)
notes_text.pack(padx=10, pady=5, fill=tk.BOTH)
notes_text.insert(tk.END, "266 top")

tk.Button(root, text="Создать видео из аудио", command=open_video_creator).pack(pady=5)

root.mainloop()
