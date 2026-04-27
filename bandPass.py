#Visualization of band pass filter using fourier transform

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt


fs = 44100          
blocksize = 2048    
input_device = None  
output_device = None 


f_low = 80    
f_high = 1500   


a_hp = 1 / (1 + (2 * np.pi * f_low / fs))
a_lp = (2 * np.pi * f_high / fs) / (1 + (2 * np.pi * f_high / fs))


hp_x1, hp_y1 = 0.0, 0.0
hp_x2, hp_y2 = 0.0, 0.0
lp_y1 = 0.0
lp_y2 = 0.0

plot_buffer = np.zeros(blocksize)


def apply_standard_bpf(x):
    global hp_x1, hp_y1, hp_x2, hp_y2
    global lp_y1, lp_y2
    
    y = np.zeros_like(x)

    for n in range(len(x)):
        out_hp1 = a_hp * (hp_y1 + x[n] - hp_x1)
        out_hp2 = a_hp * (hp_y2 + out_hp1 - hp_x2)
        
        hp_x1, hp_y1 = x[n], out_hp1
        hp_x2, hp_y2 = out_hp1, out_hp2

        out_lp1 = lp_y1 + a_lp * (out_hp2 - lp_y1)
        out_lp2 = lp_y2 + a_lp * (out_lp1 - lp_y2)
        
        lp_y1 = out_lp1
        lp_y2 = out_lp2

        y[n] = out_lp2

    return y

def callback(indata, outdata, frames, time_info, status):
    global plot_buffer
    if status:
        print(status)

    processed_audio = apply_standard_bpf(indata[:, 0])

    outdata[:, 0] = processed_audio
    outdata[:, 1] = processed_audio
    
    plot_buffer = processed_audio


plt.ion() 
fig, ax = plt.subplots(figsize=(10, 5))
xf = np.fft.rfftfreq(blocksize, 1/fs)
line, = ax.plot(xf, np.zeros(len(xf)), color='#00aaff', lw=2)

ax.set_xscale('log')
ax.set_xlim(50, 2000) 
ax.set_ylim(0, 0.05)  
ax.set_title("Standard Band-Pass Filter (1 Stage Per Side)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.grid(True, which="both", alpha=0.3)


ax.axvline(f_low, color='yellow', linestyle='--', alpha=0.8, label=f'Left Wall / HPF ({f_low}Hz)')
ax.axvline(f_high, color='cyan', linestyle='--', alpha=0.8, label=f'Right Wall / LPF ({f_high}Hz)')
ax.legend()


try:
    with sd.Stream(channels=(1, 2), samplerate=fs, blocksize=blocksize, callback=callback):
        print(f"Running Standard Band-Pass: {f_low}Hz - {f_high}Hz")
        
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