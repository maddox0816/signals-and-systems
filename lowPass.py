import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# -----------------------------
# 1. INITIAL CONFIGURATION
# -----------------------------
fs = 44100          
blocksize = 2048    
input_device = None  
output_device = None 

# -----------------------------
# 2. FILTER PARAMETERS
# -----------------------------
# Band-Pass cutoffs
f_low = 80    
f_high = 400

# NEW: Extra Final Low-Pass Cutoff
# Setting this to 250Hz to actively cut into the 200-300Hz band
f_final_lp = 100

# Calculate filter coefficients
a_hp = 1 / (1 + (2 * np.pi * f_low / fs))
a_lp = (2 * np.pi * f_high / fs) / (1 + (2 * np.pi * f_high / fs))
a_final_lp = (2 * np.pi * f_final_lp / fs) / (1 + (2 * np.pi * f_final_lp / fs))

# State Variables for High-Pass (Stage 1 & 2)
hp_x1, hp_y1 = 0.0, 0.0
hp_x2, hp_y2 = 0.0, 0.0

# State Variables for the first Low-Pass (Stage 1 & 2)
lp_y1 = 0.0
lp_y2 = 0.0

# NEW: State Variables for Final Low-Pass (Stage 1 & 2)
final_lp_y1 = 0.0
final_lp_y2 = 0.0

# Buffer to share data
plot_buffer = np.zeros(blocksize)

# -----------------------------
# 3. FILTER LOGIC
# -----------------------------
def apply_filters(x):
    global hp_x1, hp_y1, hp_x2, hp_y2 
    global lp_y1, lp_y2
    global final_lp_y1, final_lp_y2
    
    y = np.zeros_like(x)

    for n in range(len(x)):
        # --- STAGE 1: HIGH-PASS (Remove below 200Hz) ---
        out_hp1 = a_hp * (hp_y1 + x[n] - hp_x1)
        out_hp2 = a_hp * (hp_y2 + out_hp1 - hp_x2)
        
        hp_x1, hp_y1 = x[n], out_hp1
        hp_x2, hp_y2 = out_hp1, out_hp2

        # --- STAGE 2: FIRST LOW-PASS (Remove above 300Hz) ---
        out_lp1 = lp_y1 + a_lp * (out_hp2 - lp_y1)
        out_lp2 = lp_y2 + a_lp * (out_lp1 - lp_y2)

        lp_y1 = out_lp1
        lp_y2 = out_lp2

        # --- STAGE 3: FINAL LOW-PASS (Remove above 250Hz) ---
        out_final1 = final_lp_y1 + a_final_lp * (out_lp2 - final_lp_y1)
        out_final2 = final_lp_y2 + a_final_lp * (out_final1 - final_lp_y2)
        
        final_lp_y1 = out_final1
        final_lp_y2 = out_final2

        #run low pass again to get sharper cutoff
        out_final2 = lp_y1 + a_lp * (out_final2 - lp_y1)
        out_final2 = lp_y2 + a_lp * (out_final2 - lp_y2)

        final_lp_y1 = out_final2
        final_lp_y2 = out_final2

        # Final filtered sample
        y[n] = out_final2

    return y

# -----------------------------
# 4. AUDIO CALLBACK
# -----------------------------
def callback(indata, outdata, frames, time_info, status):
    global plot_buffer
    if status:
        print(status)

    # Process audio through all 3 stages
    processed_audio = apply_filters(indata[:, 0])

    # Output to speakers
    outdata[:, 0] = processed_audio
    outdata[:, 1] = processed_audio
    
    # Send to the visualizer buffer
    plot_buffer = processed_audio

# -----------------------------
# 5. VISUALIZER SETUP
# -----------------------------
plt.ion() 
fig, ax = plt.subplots(figsize=(10, 5))
xf = np.fft.rfftfreq(blocksize, 1/fs)
line, = ax.plot(xf, np.zeros(len(xf)), color='#ff00ff', lw=2)

ax.set_xscale('log')
ax.set_xlim(50, 2000) # Zoomed in even closer
ax.set_ylim(0, 0.05)  
ax.set_title("Band-Pass -> Final Low-Pass Chain")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.grid(True, which="both", alpha=0.3)

# Draw lines showing our cutoffs
ax.axvline(f_low, color='yellow', linestyle='--', alpha=0.7, label=f'BPF High-Pass ({f_low}Hz)')
ax.axvline(f_high, color='cyan', linestyle='--', alpha=0.7, label=f'BPF Low-Pass ({f_high}Hz)')
ax.axvline(f_final_lp, color='red', linestyle=':', alpha=0.9, lw=2, label=f'Final Low-Pass ({f_final_lp}Hz)')
ax.legend()

# -----------------------------
# 6. RUN LOOP
# -----------------------------
try:
    with sd.Stream(channels=(1, 2), samplerate=fs, blocksize=blocksize, callback=callback):
        print("Audio Chain Running: HPF(200) -> LPF(300) -> Final_LPF(250)")
        
        while True:
            # Calculate FFT
            mag = (np.abs(np.fft.rfft(plot_buffer)) / blocksize) * 5
            
            line.set_ydata(mag)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01) 

except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")