import matplotlib.pyplot as plt

import seaborn as sbs
import pandas as pd
import numpy as np

my_dpi = 96
size_px = (1300, 500)

plt.figure(figsize=(size_px[0] / my_dpi, size_px[1] / my_dpi), dpi=my_dpi)
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

x = np.arange(0, 7, 0.01)
    
plt.subplot(2, 1, 1)
t = "Lorem fistrum amatomaa te voy a borrar el cerito está la cosa muy malar ahorarr a gramenawer pupita hasta luego Lucas la caidita de la pradera mamaar. Ese hombree benemeritaar torpedo mamaar caballo blanco caballo negroorl fistro al ataquerl diodenoo. Tiene musho peligro ese que llega fistro fistro benemeritaar."
txt = plt.text(0.01, 0.99, t, fontsize=18, style='oblique', ha='left', va='top', wrap=True)
plt.axis('off')
txt._get_wrap_line_width = lambda: size_px[0] - (size_px[0] * 0.29)  # wrap to 600 screen pixels

plt.subplot(2, 2, 3)
plt.plot(x, np.cos(x))
    
plt.subplot(2, 2, 4)
plt.plot(x, np.sin(x)*np.cos(x))

plt.savefig("asd.pdf")