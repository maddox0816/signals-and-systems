#Visualization of high pass filter using fourier transform

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt


fs = 44100          
blocksize = 2048    
input_device = None  
output_device = None 


f_bp_low = 200    
f_bp_high = 1200


f_final_hp = 500 


a_bp_hp = 1 / (1 + (2 * np.pi * f_bp_low / fs))
a_bp_lp = (2 * np.pi * f_bp_high / fs) / (1 + (2 * np.pi * f_bp_high / fs))
a_final_hp = 1 / (1 + (2 * np.pi * f_final_hp / fs))

bp_hp_x1, bp_hp_y1 = 0.0, 0.0
bp_hp_x2, bp_hp_y2 = 0.0, 0.0

bp_lp_y1 = 0.0
bp_lp_y2 = 0.0

final_hp_x1, final_hp_y1 = 0.0, 0.0
final_hp_x2, final_hp_y2 = 0.0, 0.0

plot_buffer = np.zeros(blocksize)


def apply_chain(x):
    global bp_hp_x1, bp_hp_y1, bp_hp_x2, bp_hp_y2 
    global bp_lp_y1, bp_lp_y2
    global final_hp_x1, final_hp_y1, final_hp_x2, final_hp_y2
    
    y = np.zeros_like(x)

    for n in range(len(x)):
        out_bp_hp1 = a_bp_hp * (bp_hp_y1 + x[n] - bp_hp_x1)
        out_bp_hp2 = a_bp_hp * (bp_hp_y2 + out_bp_hp1 - bp_hp_x2)
        
        bp_hp_x1, bp_hp_y1 = x[n], out_bp_hp1
        bp_hp_x2, bp_hp_y2 = out_bp_hp1, out_bp_hp2

        out_bp_lp1 = bp_lp_y1 + a_bp_lp * (out_bp_hp2 - bp_lp_y1)
        out_bp_lp2 = bp_lp_y2 + a_bp_lp * (out_bp_lp1 - bp_lp_y2)

        bp_lp_y1 = out_bp_lp1
        bp_lp_y2 = out_bp_lp2

        out_final_hp1 = a_final_hp * (final_hp_y1 + out_bp_lp2 - final_hp_x1)
        out_final_hp2 = a_final_hp * (final_hp_y2 + out_final_hp1 - final_hp_x2)
        
        final_hp_x1, final_hp_y1 = out_bp_lp2, out_final_hp1
        final_hp_x2, final_hp_y2 = out_final_hp1, out_final_hp2

        y[n] = out_final_hp2

    return y


def callback(indata, outdata, frames, time_info, status):
    global plot_buffer
    if status:
        print(status)

    processed_audio = apply_chain(indata[:, 0])

    outdata[:, 0] = processed_audio
    outdata[:, 1] = processed_audio
    
    plot_buffer = processed_audio


plt.ion() 
fig, ax = plt.subplots(figsize=(10, 5))
xf = np.fft.rfftfreq(blocksize, 1/fs)
line, = ax.plot(xf, np.zeros(len(xf)), color='#ffaa00', lw=2)

ax.set_xscale('log')
ax.set_xlim(50, 1000) 
ax.set_ylim(0, 0.05)  
ax.set_title("Chain: Band-Pass (200-400Hz) -> Steep High-Pass (250Hz)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.grid(True, which="both", alpha=0.3)


ax.axvline(f_bp_low, color='yellow', linestyle='--', alpha=0.6, label=f'BPF Bottom ({f_bp_low}Hz)')
ax.axvline(f_bp_high, color='cyan', linestyle='--', alpha=0.6, label=f'BPF Top ({f_bp_high}Hz)')
ax.axvline(f_final_hp, color='red', linestyle=':', alpha=0.9, lw=2, label=f'Final Steep HPF ({f_final_hp}Hz)')
ax.legend()

try:
    with sd.Stream(channels=(1, 2), samplerate=fs, blocksize=blocksize, callback=callback):
        print(f"Running Chain: BPF({f_bp_low}-{f_bp_high}Hz) -> HPF({f_final_hp}Hz)")
        
        while True:
            mag = (np.abs(np.fft.rfft(plot_buffer)) / blocksize) * 5
            
            line.set_ydata(mag)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01) 

except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")