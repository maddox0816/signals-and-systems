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
# --- Stage A: Band-Pass Filter ---
f_bp_low = 200    
f_bp_high = 1200

# --- Stage B: Final Steep High-Pass Filter ---
# Set to 500Hz to aggressively cut out the bottom chunk of the Band-Pass
f_final_hp = 500 

# Calculate filter coefficients
a_bp_hp = 1 / (1 + (2 * np.pi * f_bp_low / fs))
a_bp_lp = (2 * np.pi * f_bp_high / fs) / (1 + (2 * np.pi * f_bp_high / fs))
a_final_hp = 1 / (1 + (2 * np.pi * f_final_hp / fs))

# State Variables for the Band-Pass (HPF component)
bp_hp_x1, bp_hp_y1 = 0.0, 0.0
bp_hp_x2, bp_hp_y2 = 0.0, 0.0

# State Variables for the Band-Pass (LPF component)
bp_lp_y1 = 0.0
bp_lp_y2 = 0.0

# State Variables for the Final Steep High-Pass
final_hp_x1, final_hp_y1 = 0.0, 0.0
final_hp_x2, final_hp_y2 = 0.0, 0.0

# Buffer to share data with the visualizer
plot_buffer = np.zeros(blocksize)

# -----------------------------
# 3. FILTER LOGIC
# -----------------------------
def apply_chain(x):
    # Bring in all our dedicated memory states
    global bp_hp_x1, bp_hp_y1, bp_hp_x2, bp_hp_y2 
    global bp_lp_y1, bp_lp_y2
    global final_hp_x1, final_hp_y1, final_hp_x2, final_hp_y2
    
    y = np.zeros_like(x)

    for n in range(len(x)):
        # ==========================================
        # PART 1: THE BAND-PASS FILTER
        # ==========================================
        
        # --- A. BPF High-Pass (Remove below 200Hz) ---
        out_bp_hp1 = a_bp_hp * (bp_hp_y1 + x[n] - bp_hp_x1)
        out_bp_hp2 = a_bp_hp * (bp_hp_y2 + out_bp_hp1 - bp_hp_x2)
        
        bp_hp_x1, bp_hp_y1 = x[n], out_bp_hp1
        bp_hp_x2, bp_hp_y2 = out_bp_hp1, out_bp_hp2

        # --- B. BPF Low-Pass (Remove above 400Hz) ---
        out_bp_lp1 = bp_lp_y1 + a_bp_lp * (out_bp_hp2 - bp_lp_y1)
        out_bp_lp2 = bp_lp_y2 + a_bp_lp * (out_bp_lp1 - bp_lp_y2)

        bp_lp_y1 = out_bp_lp1
        bp_lp_y2 = out_bp_lp2

        # ==========================================
        # PART 2: THE FINAL STEEP HIGH-PASS FILTER
        # ==========================================
        # This takes the output of the BPF (out_bp_lp2) and runs it 
        # through high-pass math twice for a steeper cut.
        
        out_final_hp1 = a_final_hp * (final_hp_y1 + out_bp_lp2 - final_hp_x1)
        out_final_hp2 = a_final_hp * (final_hp_y2 + out_final_hp1 - final_hp_x2)
        
        final_hp_x1, final_hp_y1 = out_bp_lp2, out_final_hp1
        final_hp_x2, final_hp_y2 = out_final_hp1, out_final_hp2

        # Final filtered sample to speakers
        y[n] = out_final_hp2

    return y

# -----------------------------
# 4. AUDIO CALLBACK
# -----------------------------
def callback(indata, outdata, frames, time_info, status):
    global plot_buffer
    if status:
        print(status)

    processed_audio = apply_chain(indata[:, 0])

    outdata[:, 0] = processed_audio
    outdata[:, 1] = processed_audio
    
    plot_buffer = processed_audio

# -----------------------------
# 5. VISUALIZER SETUP
# -----------------------------
plt.ion() 
fig, ax = plt.subplots(figsize=(10, 5))
xf = np.fft.rfftfreq(blocksize, 1/fs)
line, = ax.plot(xf, np.zeros(len(xf)), color='#ffaa00', lw=2)

ax.set_xscale('log')
ax.set_xlim(50, 1000) # Zoomed in specifically on the low-mid frequencies
ax.set_ylim(0, 0.05)  
ax.set_title("Chain: Band-Pass (200-400Hz) -> Steep High-Pass (250Hz)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.grid(True, which="both", alpha=0.3)

# Draw lines showing our cutoffs
ax.axvline(f_bp_low, color='yellow', linestyle='--', alpha=0.6, label=f'BPF Bottom ({f_bp_low}Hz)')
ax.axvline(f_bp_high, color='cyan', linestyle='--', alpha=0.6, label=f'BPF Top ({f_bp_high}Hz)')
ax.axvline(f_final_hp, color='red', linestyle=':', alpha=0.9, lw=2, label=f'Final Steep HPF ({f_final_hp}Hz)')
ax.legend()

# -----------------------------
# 6. RUN LOOP
# -----------------------------
try:
    with sd.Stream(channels=(1, 2), samplerate=fs, blocksize=blocksize, callback=callback):
        print(f"Running Chain: BPF({f_bp_low}-{f_bp_high}Hz) -> HPF({f_final_hp}Hz)")
        
        while True:
            # Multiplier here boosts the visual wave so you can see it easily
            mag = (np.abs(np.fft.rfft(plot_buffer)) / blocksize) * 5
            
            line.set_ydata(mag)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01) 

except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")