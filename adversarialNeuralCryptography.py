import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import time



"""
hyper parameters
"""
N = 16
PTEXT_SIZE = 16
KEY_SIZE = 16
CTEXT_SIZE = 16

CLIP_VALUE = 1
LEARNING_RATE = 0.0008
BATCH_SIZE = 256
MAX_TRAINING_LOOPS = 100000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "./adversarial_neural_cryptography_model_and_optimizer"

EVE_ONE_BIT_WRONG_THRESH = 0.97
BOB_ONE_BIT_WRONG_THRESH = 0.0025

LOOPS_PER_PRINT = 100   # every 100 loops print one time



class Model(nn.Module):
    """
    the model alice, bob and eve.
    1 linear + 4 Conv1d.
    """
    def __init__(self, text_size, key_size = None):
        super(Model, self).__init__()
        self.linear = self.linear_init(text_size, key_size)
        self.conv1 = nn.Conv1d(1, 2, 4, stride=1, padding=2)
        self.conv2 = nn.Conv1d(2, 4, 2, stride=2)
        self.conv3 = nn.Conv1d(4, 4, 1, stride=1)
        self.conv4 = nn.Conv1d(4, 1, 1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()



    def forward(self, x):
        x = x[None, :, :].transpose(0, 1)
        x = self.sigmoid(self.linear(x))
        x = self.sigmoid(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        x = self.tanh(self.conv4(x))
        x = torch.squeeze(x, 1)
        return x

    def linear_init(self, text_size, key_size):
        if key_size is not None:
            return nn.Linear(text_size + key_size, 2 * N)
        else:
            return nn.Linear(text_size, 2 * N)



def generate_data(batch_size, ptext_size, key_size):
    """
    generate data.

    :param batch_size: batch size, hyper-parameters, in this program BATCH_SIZE is 256
    :param ptext_size: ptext size, hyper-parameters, in this program PTEXT_SIZE is 16
    :param key_size: key's size, hyper-parameters, in this program KEY_SIZE is 16
    :return: ptext and key, in this program size are both [256, 16]
    """
    ptext = torch.randint(0, 2, (batch_size, ptext_size), dtype=torch.float).to(DEVICE) * 2 - 1
    key = torch.randint(0, 2, (batch_size, key_size), dtype=torch.float).to(DEVICE) * 2 - 1
    return ptext, key



def plot_wrong(eve_wrong_for_plot, bob_wrong_for_plot):
    """
    plot epoch-wrong picture

    :param eve_wrong_for_plot: a list, element is the mean of eve one bit wrong
    :param bob_wrong_for_plot: a list, element is the mean of bob one bit wrong
    :return:
    """
    plt.plot(range(1, len(eve_wrong_for_plot)+1), eve_wrong_for_plot, label='eve one bit wrong mean')
    plt.plot(range(1, len(bob_wrong_for_plot)+1), bob_wrong_for_plot, label='bob one bit wrong mean')
    plt.xlabel("Epochs")
    plt.ylabel("One Bit Wrong")
    plt.title("optimizer_bob_times: optimizer_eve_times = 1 : 2")
    plt.legend()
    plt.show()



def train():
    """
    Do the following:
    1. generate data
    2. train model
    3. finish running and save parameters if satisfing conditions
    4. print the waste of time and errors
    5. plot epochs-errors picture when finish running
    """

    # init
    eve_one_bit_wrong_mean = 2.0
    bob_one_bit_wrong_mean = 2.0

    eve_wrong_for_plot = []
    bob_wrong_for_plot = []

    alice = Model(PTEXT_SIZE, KEY_SIZE).to(DEVICE)
    bob = Model(CTEXT_SIZE, KEY_SIZE).to(DEVICE)
    eve = Model(CTEXT_SIZE).to(DEVICE)

    alice.train()
    bob.train()
    eve.train()

    optimizer_alice = optim.Adam(alice.parameters(), lr=LEARNING_RATE)
    optimizer_bob = optim.Adam(bob.parameters(), lr=LEARNING_RATE)
    optimizer_eve = optim.Adam(eve.parameters(), lr =LEARNING_RATE)

    # loss function
    bob_reconstruction_error = nn.L1Loss()
    eve_reconstruction_error = nn.L1Loss()

    for i in range(MAX_TRAINING_LOOPS):

        start_time = time.time()

        # if satisfy conditions, finish running and save parameters.
        if eve_one_bit_wrong_mean > EVE_ONE_BIT_WRONG_THRESH and bob_one_bit_wrong_mean < BOB_ONE_BIT_WRONG_THRESH:
            print()
            print("Satisfing Conditions.")

            # 保存model参数、 optimizer参数 和 eve_one_bit_wrong_mean、 bob_one_bit_wrong_mean
            torch.save({
                'Alice_state_dict': alice.state_dict(),
                'Bob_state_dict': bob.state_dict(),
                'Eve_state_dict': eve.state_dict(),
                'optimizer_alice_state_dict': optimizer_alice.state_dict(),
                'optimizer_bob_state_dict': optimizer_bob.state_dict(),
                'optimizer_eve_state_dict': optimizer_eve.state_dict(),
                'bob_one_bit_wrong_mean': bob_one_bit_wrong_mean,
                'eve_one_bit_wrong_mean': eve_one_bit_wrong_mean
            }, SAVE_PATH)

            print('Saved the parameters successfully.')
            break

        # train alice_bob : train eve = 1 : 2
        for network, num_minibatch in {'alice_bob': 1, 'eve': 2}.items():

            for minibatch in range(num_minibatch):

                ptext, key = generate_data(BATCH_SIZE, PTEXT_SIZE, KEY_SIZE)

                ctext = alice(torch.cat((ptext, key), 1).float())
                ptext_eve = eve(ctext)

                if network == 'alice_bob':

                    ptext_bob = bob(torch.cat((ctext, key), 1).float())

                    error_bob = bob_reconstruction_error(ptext_bob, ptext)
                    error_eve = eve_reconstruction_error(ptext_eve, ptext)
                    alice_bob_loss = error_bob + (1.0 - error_eve ** 2)

                    optimizer_alice.zero_grad()
                    optimizer_bob.zero_grad()
                    alice_bob_loss.backward()
                    nn.utils.clip_grad_value_(alice.parameters(), CLIP_VALUE)
                    nn.utils.clip_grad_value_(bob.parameters(), CLIP_VALUE)
                    optimizer_alice.step()
                    optimizer_bob.step()

                elif network == 'eve':

                    error_eve = eve_reconstruction_error(ptext_eve, ptext)

                    optimizer_eve.zero_grad()
                    error_eve.backward()
                    nn.utils.clip_grad_value_(eve.parameters(), CLIP_VALUE)
                    optimizer_eve.step()

        time_elapsed = time.time() - start_time

        bob_one_bit_wrong_mean = error_bob.cpu().detach().numpy()
        eve_one_bit_wrong_mean = error_eve.cpu().detach().numpy()

        if i % LOOPS_PER_PRINT == 0:
            print(f'Epoch: {i + 1:06d} | '
                  f'one epoch time: {time_elapsed:.3f} | '
                  f'bob one bit wrong: {bob_one_bit_wrong_mean:.4f} |'
                  f'eve one bit wrong: {eve_one_bit_wrong_mean:.4f}')

        eve_wrong_for_plot.append(eve_one_bit_wrong_mean)
        bob_wrong_for_plot.append(bob_one_bit_wrong_mean)

    plot_wrong(eve_wrong_for_plot, bob_wrong_for_plot)




if __name__ == "__main__":

    train()
