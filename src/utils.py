import numpy as np
import torch
import gzip
from matplotlib import pyplot as plt

def one_hot_encode(seq):
    seq2 = [mapping['A'] if i == 'N' else mapping[i] for i in seq]
    
    return np.eye(4)[seq2]

def preprocessing(dataset):
    # open gzip file
    dataset = dataset
    filename = f"/home/jiwon/BScProject/{dataset}.gz"

    with gzip.open(filename, 'rb') as f:
        seqs = f.readlines()
        acc = []

        for i, seq in enumerate(seqs):
            acc.append(seq.decode("utf-8").rstrip('\n'))
        
        if i == 10:
            print(seq.decode("utf-8").rstrip('\n'))
        
        print(f"Length of one sequence: {len(acc[0])}")
        print(f"Number of positive sequences: {len(acc)}")

    # one-hot encode and save np arrays
    one_hot = []
    mapping = dict(zip("ACGT", range(4)))

    for seq in acc:
        one_hot.append(one_hot_encode(seq))
        one_hot = np.array(one_hot)

def matrix_to_seq(seq):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    result_seq = ''
    
    for base_vector in seq:
        base_vector = list(base_vector)
        i = base_vector.index(max(base_vector))
        result_seq += mapping[i]
        
    return result_seq

def sequence_generator(num_epoch, dir_name, fake):
    mat_seqs = fake[:, :, :, 0]

    f = open(f'{dir_name}Synthetic_Sequences_Epoch{num_epoch}.txt', 'w')
    str_seqs = []

    for v in mat_seqs:
        seq = matrix_to_seq(v)

        if seq[197:199] == 'AG' and seq not in str_seqs:
            f.write(seq + '\n')
            str_seqs.append(seq)

    f.close()

def sequence_generator_as_list(fake):
    mat_seqs = fake[:, :, :, 0]
    str_seqs = []

    for v in mat_seqs:
        seq = matrix_to_seq(v)

        if seq[197:199] == 'AG' and seq not in str_seqs:
            str_seqs.append(seq)

    return str_seqs

def loss_per_graph(run_name, d_loss_fake, d_loss_real, d_loss, g_loss, d_per, g_per):
    
    step = 1
    epoch = len(d_loss_fake) // step
    xplot = list(range(1, epoch + 1))
    xplot = [x * step for x in xplot]

    plt.plot(xplot, d_loss_fake, label='loss_discriminator_fake data')
    plt.plot(xplot, d_loss_real, label='loss_discriminator_real data')
    plt.plot(xplot, g_loss, label='loss_generator')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and discriminator loss')
    plt.savefig(f'/home/jiwon/BScProject/{run_name}/{run_name}_model_loss.svg', format='svg', dpi=1200)

    plt.figure()
    plt.plot(xplot, d_per, label='output_real data')
    plt.plot(xplot, g_per, label='output_fake data')
    plt.xlabel('Epochs')
    plt.ylabel('Mean output value')
    plt.legend()
    plt.title('Mean output value of discriminator')
    plt.savefig(f'/home/jiwon/BScProject/{run_name}/{run_name}_model_performance.svg', format='svg', dpi=1200)

def calculate_ppm():
    motifs = [list(line.rstrip()) for line in open('/home/jiwon/BScProject/sequence_logo/motifs_epoch_100.txt')]
    num_seqs = len(motifs)

    motifs_t = [list(x) for x in zip(*motifs)]
    ppm = []

    for i, pos in enumerate(motifs_t):
        prob = [pos.count('A')/num_seqs*100, pos.count('C')/num_seqs*100, pos.count('G')/num_seqs*100, pos.count('T')/num_seqs*100]
        ppm.append(prob)
    
    return ppm

def gen_len(run_name, num):
    plt.figure()
    plt.plot(num, linestyle='--',marker='o', label='Number of generated sequence')
    plt.xlabel('Epochs')
    plt.ylabel('Number of generated sequences')
    plt.title('Number of generated sequences (w/o BatchNorm in $D$ and $G$)')
    plt.savefig(f'/home/jiwon/BScProject/{run_name}/{run_name}_fake_len.svg', format='svg', dpi=1200)

def score_graph(run_name, score):
    with open('/home/jiwon/BScProject/ppm.txt', 'w+') as file:
        file.write(str(score))
    
    plt.figure()
    plt.plot(score, label='PPM score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('PPM score during training')
    plt.savefig(f'/home/jiwon/BScProject/{run_name}/{run_name}_motif_score.svg', format='svg', dpi=1200)

def editDistance(v, w):
    n, m = len(v), len(w)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Initialize the edit graph with v0i=i, vj0=j
    for i in range(m + 1):
        d[0][i] = i
    for j in range(n + 1):
        d[j][0] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            # If two characters are the same, take the diagonal one
            if v[i - 1] == w[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min([d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1])
    
    # The last element is the edit distance of v and w
    return d[-1][-1]

def construct_matrix(v, w):
    n, m = len(v), len(w)
    # Make matrices of score and backtracking pointers
    s = [[0] * (m + 1) for _ in range(n + 1)]
    b = [[0] * m for _ in range(n)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Recurrence for computing LCS
            # LCS x dp table
            if v[i - 1] == w[j - 1]:
                s[i][j] = max([s[i - 1][j], s[i][j - 1], s[i - 1][j - 1] + 1])
            else:
                s[i][j] = max([s[i - 1][j], s[i][j - 1], s[i - 1][j - 1]])
            # Maintain backtracking pointers
            if s[i][j] == s[i - 1][j]:
                b[i - 1][j - 1] = "up"
            else:
                if s[i][j] == s[i][j - 1]:
                    b[i - 1][j - 1] = "left"
                else:
                    b[i - 1][j - 1] = "diag"
    return b

def backtrack(b, v, i, j):
    seq = ''
    while True:
        if i == -1 or j == -1:
            return seq[::-1]

        if b[i][j] == 'up':
            i -= 1
        elif b[i][j] == "left":
            j -= 1
        elif b[i][j] == "diag":
            # Characters are matched
            seq += v[i]
            i -= 1
            j -= 1

def lcs(seq1, seq2):
    b = construct_matrix(seq1, seq2)
    res_seq = backtrack(b, seq1, len(seq1) - 1, len(seq2) - 1)
    
    return len(res_seq)

def txt_change_list(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            sequence = line.strip()
            sequences.append(sequence)
    return sequences

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    # calculate intersection
    intersection = len(set1.intersection(set2))

    # calculate union
    union = len(set1.union(set2))

    # calculate Jaccard similarity
    similarity = (intersection / union)

    return similarity

def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers

def jaccard_similarity_plot(run_name, similarity):
    plt.figure()
    plt.plot(similarity, label='Jaccard similarity')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard similarity')
    plt.ylim((0, 1))
    plt.legend()
    plt.title('Jaccard similarity during training using 2-mers')
    plt.savefig(f'/home/jiwon/BScProject/jaccard/{run_name}_jaccard.svg', format='svg', dpi=1200)