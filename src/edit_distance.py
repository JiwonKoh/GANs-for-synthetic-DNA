import utils
import random
from matplotlib import pyplot as plt

random.seed(25)

training = [line.rstrip() for line in open('/home/jiwon/BScProject/GWH_datasets/GWH_acceptors_chr1_pos.txt')][:6671]
testing = [line.rstrip() for line in open('/home/jiwon/BScProject/GWH_datasets/GWH_acceptors_chr1_pos.txt')][6671:]
synthetic = [line.rstrip() for line in open('/home/jiwon/BScProject/Edit_distance/Synthetic_Sequences_Epoch4000.txt')]

# Randomly choice 1000 seqs from training and synthetic
training = random.sample(training, 1000)
synthetic = random.sample(synthetic, 1000)

# Calculate edit distance in pairwise manner
ed_testing = dict()
for i, seq in enumerate(training):
    ed = utils.editDistance(seq, testing[i])
    if ed in ed_testing:
        ed_testing[ed] += 1
    else:
        ed_testing[ed] = 1

ed_synthetic = dict()
for i, seq in enumerate(training):
    ed = utils.editDistance(seq, synthetic[i])
    if ed in ed_synthetic:
        ed_synthetic[ed] += 1
    else:
        ed_synthetic[ed] = 1

print(ed_testing)
print(ed_synthetic)

# Sort the dictionaries by keys
sorted_ed_testing = dict(sorted(ed_testing.items()))
sorted_ed_synthetic = dict(sorted(ed_synthetic.items()))

# Extract keys and values from the dictionaries
x1, y1 = zip(*sorted_ed_testing.items())
x2, y2 = zip(*sorted_ed_synthetic.items())

plt.figure(figsize=(20, 5))
# Plot the bar graphs with transparency
plt.bar(x1, y1, alpha=0.5, label='Testing data')
plt.bar(x2, y2, alpha=0.5, label='Synthetic data')

# Sort the x axis
plt.xticks(sorted(list(set(x1) | set(x2))))
plt.xticks(rotation=60)

# Add labels and legend
plt.xlabel('Edit distance', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.title('Distribution of edit distance', fontsize=18)
plt.legend(fontsize=12)

# Save plot
plt.savefig(f'/home/jiwon/BScProject/edit_distance/edit_distance_distribution2.svg', format='svg', dpi=1200, bbox_inches = "tight")