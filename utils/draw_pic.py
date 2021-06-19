import csv
from matplotlib import pyplot as plt
import mpl_toolkits.axisartist as axisartist

fig = plt.figure(dpi=128, figsize=(8, 10))
ax = axisartist.Subplot(fig, 111)
fig.add_axes(ax)
filename = './data_results/fig5.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    header_row = next(reader)
    highs1 = []
    highs2 = []
    highs3 = []
    highs4 = []

    for row in reader:
        highs1.append(float(row[1]))
        highs2.append(float(row[2]))
        highs3.append(float(row[3]))
        highs4.append(float(row[4]))
    print(highs1)
    print(highs2)
    print(highs3)
    print(highs4)

l1, = plt.plot(highs1, c='gray')
l2, = plt.plot(highs2, c='blue')
l3, = plt.plot(highs3, c='green')
l4, = plt.plot(highs4, c='red')
plt.legend(handles=[l4, l3, l2, l1], labels=['8 MSRBs', '6 MSRBs', '4 MSRBs', '2 MSRBs'], loc='upper right')
plt.title('MR = 0.25', fontsize=24, fontweight='heavy')
plt.xlabel('Epoch Number in Training', fontsize=16)
plt.ylabel('PSNR(dB)', fontsize=16)
ax.axis['top'].set_visible(False)
ax.axis['right'].set_visible(False)
ax.axis["bottom"].set_axisline_style("-|>", size=1.0)
ax.axis["left"].set_axisline_style("-|>", size=1.0)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

plt.show()
