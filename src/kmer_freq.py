import numpy as np
from matplotlib import pyplot as plt

# Load file
motifs1 = [list(line.rstrip()) for line in open('/home/jiwon/BScProject/sequence_logo/motifs_epoch_100.txt')]
motifs2 = [list(line.rstrip()) for line in open('/home/jiwon/BScProject/sequence_logo/motifs_real.txt')]

two_mers1 = dict()

for motif in motifs1:
    for i in range(len(motif) - 1):
        two_mer = ''.join(motif[i:i+2])

        if two_mer in two_mers1:
            two_mers1[two_mer] += 1
        else:
            two_mers1[two_mer] = 1

sorted_two_mers1 = dict(sorted(two_mers1.items()))

two_mers2 = dict()

for motif in motifs2:
    for i in range(len(motif) - 1):
        two_mer = ''.join(motif[i:i+2])

        if two_mer in two_mers2:
            two_mers2[two_mer] += 1
        else:
            two_mers2[two_mer] = 1

sorted_two_mers2 = dict(sorted(two_mers2.items()))

# Extract keys and values from the dictionaries
x1, y1 = zip(*sorted_two_mers1.items())
x2, y2 = zip(*sorted_two_mers2.items())

# Plot the bar graphs with transparency
plt.bar(np.arange(16), y1, 0.4, label='Synthetic data at epoch 4000')
plt.bar(np.arange(16)+0.4, y2, 0.4, label='Real data')
# Sort the x axis
plt.xticks(np.arange(16) + 0.4/2, sorted(list(set(x1))))
plt.xticks(rotation=90)
plt.ylim((0, 55000))

# Add labels and legend
plt.xlabel('2-mers')
plt.ylabel('Counts')
plt.title('2-mer frequency of real motifs and synthetic motifs')
plt.legend()

# Save plot
plt.savefig(f'/home/jiwon/BScProject/k_mer_freq/2-kmer_freq_real_vs_epoch4000_scaled.svg', format='svg', dpi=1200, bbox_inches = "tight")
