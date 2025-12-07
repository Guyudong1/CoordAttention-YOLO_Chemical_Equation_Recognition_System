import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

latex_str = r"$2 H_2O_2 \xrightarrow{MnO_2} 2 H_2O + O_2$"

plt.figure(figsize=(8, 2))
plt.text(0.1, 0.5, latex_str, fontsize=24)
plt.axis('off')

# 保存为矢量 PDF 或 SVG
plt.savefig("eq.pdf", bbox_inches='tight')
# plt.savefig("eq.svg", bbox_inches='tight')

plt.show()
