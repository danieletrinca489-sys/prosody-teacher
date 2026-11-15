import gradio as gr
import whisper
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import subprocess

model = whisper.load_model("base")

def convert_to_wav(input_path):
    output_path = "temp_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return output_path

def get_intonation_arrow(pitch_segment):
    valid_pitches = pitch_segment[~np.isnan(pitch_segment)]
    if len(valid_pitches) < 3:
        return "→"
    n = len(valid_pitches)
    first_half = np.mean(valid_pitches[:n//2])
    second_half = np.mean(valid_pitches[n//2:])
    diff = second_half - first_half
    if diff > 10:
        return "↗️"
    elif diff < -10:
        return "↘️"
    else:
        return "→"

def analizza_voce(audio_path):
    if audio_path is None:
        return None, "⚠️ Registra o carica un file audio prima di avviare l'analisi."

    wav_filename = convert_to_wav(audio_path)

    result = model.transcribe(wav_filename, word_timestamps=True)
    words_data = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words_data.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"]
            })

    if not words_data:
        return None, "⚠️ Nessuna parola riconosciuta."

    sound = parselmouth.Sound(wav_filename)
    pitch = sound.to_pitch()
    pitch_times = pitch.xs()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan

    intensity = sound.to_intensity()
    int_times = intensity.xs()
    int_values = intensity.values.T.flatten()

    g_min = np.nanpercentile(pitch_values, 10)
    g_max = np.nanpercentile(pitch_values, 90)
    p_range = max(g_max - g_min, 1.0)
    mean_int = np.mean(int_values)

    for item in words_data:
        s, e = item["start"], item["end"]
        p_mask = (pitch_times >= s) & (pitch_times <= e)
        i_mask = (int_times >= s) & (int_times <= e)
        w_p = pitch_values[p_mask]
        w_i = int_values[i_mask]

        mean_p = np.nanmean(w_p) if np.any(~np.isnan(w_p)) else g_min
        mean_v = np.mean(w_i) if len(w_i) > 0 else mean_int

        item["norm_y"] = np.clip((mean_p - g_min) / p_range, 0.1, 0.9)
        item["stressed"] = mean_v > (mean_int + 1.2)
        item["arrow"] = get_intonation_arrow(w_p)

    total_duration = words_data[-1]["end"] + 0.3
    fig, ax = plt.subplots(figsize=(max(11, total_duration * 3), 5.5))

    for y_line in [1.25, 1.75, 2.25]:
        ax.axhline(y_line, color='#E0E0E0', linestyle='--', linewidth=1)
    for y_line in [0.25, 0.75, 1.25]:
        ax.axhline(y_line, color='#E0E0E0', linestyle='--', linewidth=1)

    ax.axhline(1.5, color='#333333', linestyle='-', linewidth=1.8)

    ax.text(-0.08, 2.40, "MODELLO IDEALE", fontsize=11, fontweight='bold', color='#1E88E5')
    ax.text(-0.08, 1.38, "LA TUA VOCE", fontsize=11, fontweight='bold', color='#E6194B')

    for item in words_data:
        start = item["start"]
        duration = max(item["end"] - item["start"], 0.08)
        y_pos = 0.45 + (item["norm_y"] * 0.65)
        
        block_h = 0.22 if item["stressed"] else 0.16
        color = '#E6194B' if item["stressed"] else '#F06292'
        txt = item["word"].upper() if item["stressed"] else item["word"].lower()

        rect = patches.FancyBboxPatch(
            (start, y_pos - block_h/2), duration, block_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color, edgecolor='none', zorder=3
        )
        ax.add_patch(rect)
        ax.text(start + duration/2, y_pos, txt, color='white', fontweight='bold', fontsize=10, ha='center', va='center', zorder=4)
        ax.text(start + duration/2, 1.25, item["arrow"], color='#E6194B', fontsize=15, fontweight='bold', ha='center', va='center')

        ideal_y = 1.75 + (0.3 if item["stressed"] else 0.0)
        ideal_rect = patches.FancyBboxPatch(
            (start, ideal_y - 0.1), duration, 0.2,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor='#1E88E5' if item["stressed"] else '#64B5F6', edgecolor='none', zorder=3
        )
        ax.add_patch(ideal_rect)
        ax.text(start + duration/2, ideal_y, txt, color='white', fontweight='bold', fontsize=10, ha='center', va='center', zorder=4)
        ax.text(start + duration/2, 2.40, "↘️" if item["stressed"] else "→", color='#1E88E5', fontsize=15, fontweight='bold', ha='center', va='center')

    ax.set_xlim(-0.15, total_duration)
    ax.set_ylim(0, 2.65)
    ax.set_xlabel("Tempo (secondi)", fontsize=10)
    ax.set_yticks([])
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    frase_riconosciuta = " ".join([w["word"] for w in words_data])
    return fig, f"✅ Frase analizzata: «{frase_riconosciuta}»"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 Laboratorio di Prosodia Inglese")
    gr.Markdown("Registra la tua voce per visualizzare stress e intonazione.")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Registra o carica la frase")
    
    btn = gr.Button("Analizza Prosodia", variant="primary")
    status_output = gr.Textbox(label="Stato dell'analisi", interactive=False)
    plot_output = gr.Plot(label="Partitura Prosodica Comparativa")
    
btn.click(fn=analizza_voce, inputs=audio_input, outputs=[plot_output, status_output])

if __name__ == "__main__":
    demo.launch()
