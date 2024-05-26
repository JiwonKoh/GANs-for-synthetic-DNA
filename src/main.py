import os, argparse
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from matplotlib import pyplot as plt
import model as m
import utils
import random

# Set random seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(25)

# Device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Now using {} devices".format(device))

class GAN():
    def __init__(self, run_name='test1'):
        
        # Parameters
        self.num_epoch = 100
        self.batch_size = 64
        self.seq_len = 398
        self.data_dir = "/home/jiwon/BScProject/GWH_datasets/GWH_acceptors_chr1.npy"
        self.run_name = run_name
        self.noise_size = 100
        self.sample_size = 1

        # Load data
        acc = np.load(self.data_dir)
        acc = torch.from_numpy(acc).float()
        self.data_loader = torch.utils.data.DataLoader(dataset=acc, batch_size=self.batch_size, drop_last=True, shuffle=True)
        print(f"The number of mini-batch: {len(self.data_loader)}")

    def build_model(self):
        # Initialize Generator/Discriminator
        self.d = m.Discriminator()
        self.g = m.Generator()

        # Device setting
        self.d = self.d.to(device)
        self.g = self.g.to(device)

        self.criterion = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=0.00002)
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=0.00001)

    def train_model(self):

        d_epoch_loss_fake = []
        d_epoch_loss_real = []
        d_epoch_loss = []
        g_epoch_loss = []
        d_epoch_per = []
        g_epoch_per = []
        ppm_score = []
        # For Jaccard similarity
        real_motifs = [str(line.rstrip()) for line in open('/home/jiwon/BScProject/sequence_logo/motifs_real.txt')]
        real_motifs = random.sample(real_motifs, self.sample_size)
        jaccard = []
        fake_len = []

        os.makedirs(f"/home/jiwon/BScProject/{self.run_name}/", exist_ok=True)

        for epoch in range(int(self.num_epoch)):
        
            d_losses_fake = 0
            d_losses_real = 0
            d_losses = 0
            g_losses = 0
            d_performances = 0
            g_performances = 0

            for i, seqs in enumerate(self.data_loader):
                real_label = torch.full((self.batch_size, 1), 1, dtype=torch.float32).to(device)
                fake_label = torch.full((self.batch_size, 1), 0, dtype=torch.float32).to(device)

                real_seqs = seqs.reshape(self.batch_size, -1).to(device)

                """
                Train Generator
                """

                # Initialize gradient
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                # Make fake sequences with generator and noise vector z
                z = torch.randn(self.batch_size, self.noise_size).to(device)
                fake_seqs = self.g(z)

                g_loss = self.criterion(self.d(fake_seqs), real_label)
                g_loss.backward()
                self.g_optimizer.step()

                """
                Train Discriminator
                """

                # Initialize gradient
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()

                z = torch.randn(self.batch_size, self.noise_size).to(device)
                fake_seqs = self.g(z)

                # Calculate fake & real loss with generated images above & real images
                fake_loss = self.criterion(self.d(fake_seqs), fake_label)
                real_loss = self.criterion(self.d(real_seqs), real_label)
                d_loss = (fake_loss + real_loss) / 2

                d_loss.backward()
                self.d_optimizer.step()

                d_performance = self.d(real_seqs).mean() 
                g_performance = self.d(fake_seqs).mean() 

                d_losses_fake += fake_loss.cpu().detach().numpy().mean()
                d_losses_real += real_loss.cpu().detach().numpy().mean()
                d_losses += d_loss.cpu().detach().numpy().mean()
                g_losses += g_loss.cpu().detach().numpy().mean()
                d_performances += d_performance.cpu().detach().numpy()
                g_performances += g_performance.cpu().detach().numpy()
            
            with torch.no_grad():
                # Switch model to eval mode
                self.g.eval()
                
                # if epoch+1 == 100:
                random_noise = torch.randn(self.sample_size, self.noise_size).to(device)
                fake = self.g(random_noise).detach()

                fake = fake.reshape(self.sample_size, 398, 4, 1)
                # Save sequences as txt files
                # utils.sequence_generator(epoch+1, f"/home/jiwon/BScProject/{self.run_name}/", fake)
                Save sequences in list
                fake_seqs = utils.sequence_generator_as_list(fake)
                fake_seqs_len = len(fake_seqs)
            """
            # Calculate ppm score 
            ppm = utils.calculate_ppm()

            total_score = 0
            for seq in fake_seqs:
                motif = seq[177:200]

                score = 1
                for i, base in enumerate(motif):
                    score *= ppm[i]['ACGT'.index(base)]

                total_score += score

            ppm_score.append(total_score/self.sample_size)
            """
            """
            # Calculate Jaccard similarity
            similarity = 0
            for seq in fake_seqs:
                fake_kmers = utils.build_kmers(seq[177:200], 2)
                for seq2 in real_motifs:
                    real_kmers = utils.build_kmers(seq2, 2)
                    similarity += utils.jaccard_similarity(fake_kmers, real_kmers)

            jaccard.append(similarity / self.sample_size**2)
            """
            d_epoch_loss_fake.append(d_losses_fake / len(self.data_loader))
            d_epoch_loss_real.append(d_losses_real / len(self.data_loader))
            d_epoch_loss.append(d_losses / len(self.data_loader))
            g_epoch_loss.append(g_losses / len(self.data_loader))

            d_epoch_per.append(d_performances / len(self.data_loader))
            g_epoch_per.append(g_performances / len(self.data_loader))
            fake_len.append(fake_seqs_len)

            print(f"Epoch {epoch+1}) D performance : {d_performance},  G performance : {g_performance}")
            print(f"Epoch {epoch+1}) D loss: {d_loss.mean()}, G loss: {g_loss.mean()}")

        # Visualization
        utils.loss_per_graph(self.run_name, d_epoch_loss_fake, d_epoch_loss_real, \
                            d_epoch_loss, g_epoch_loss, d_epoch_per, g_epoch_per)
        # Jaccard similarity
        # utils.jaccard_similarity_plot(self.run_name, jaccard)
        # Number of generated sequences
        utils.gen_len(self.run_name, fake_len)

        # Save models
        torch.save(self.g.state_dict(), f"/home/jiwon/BScProject/{self.run_name}/Generator.pt")
        torch.save(self.d.state_dict(), f"/home/jiwon/BScProject/{self.run_name}/Discriminator.pt")

def main():
    parser = argparse.ArgumentParser(description='GAN for producing DNA sequences.')
    parser.add_argument("--run_name", default= "BaP_test", help="Name for output files (checkpoint and sample dir)")
    args = parser.parse_args()
    model = GAN(run_name=args.run_name)
    model.build_model()
    model.train_model()

if __name__ == '__main__':
    main()

