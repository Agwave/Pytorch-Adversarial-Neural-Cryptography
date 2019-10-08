import torch

import adversarialNeuralCryptography as ad

LOAD_PATH = "./adversarial_neural_cryptography_model_and_optimizer"



def random_generate_ptext_and_key(ptext_size, key_size):
    """
    generate a ptext and a key for validate
    """
    ptext = torch.randint(0, 2, (1, ptext_size), dtype=torch.float).to(ad.DEVICE) * 2 - 1
    key = torch.randint(0, 2, (1, key_size), dtype=torch.float).to(ad.DEVICE) * 2 - 1
    return ptext, key



def model_load_checkpoint():
    """
    alice, bob, eve load checkpoint
    :return: a tuple: (alice, bob, eve)
    """
    checkpoint = torch.load(LOAD_PATH)

    alice = ad.Model(ad.PTEXT_SIZE, ad.KEY_SIZE)
    bob = ad.Model(ad.PTEXT_SIZE, ad.KEY_SIZE)
    eve = ad.Model(ad.PTEXT_SIZE)

    alice.load_state_dict(checkpoint['Alice_state_dict'])
    bob.load_state_dict(checkpoint['Bob_state_dict'])
    eve.load_state_dict(checkpoint['Eve_state_dict'])

    alice.to(ad.DEVICE)
    bob.to(ad.DEVICE)
    eve.to(ad.DEVICE)

    return alice, bob, eve



def validate():
    """
    generate a ptext and key and compare them to the output of the model
    :return:
    """
    ptext, key = random_generate_ptext_and_key(ad.PTEXT_SIZE, ad.KEY_SIZE)

    alice, bob, eve = model_load_checkpoint()

    ctext = alice(torch.cat((ptext, key), 1).float())

    predict_ptext_bob = bob(torch.cat((ctext, key), 1).float())
    predict_ptext_eve = eve(ctext)

    # for better print
    ptext = ptext.cpu().detach().numpy()
    predict_ptext_bob = predict_ptext_bob.cpu().detach().numpy()
    predict_ptext_eve = predict_ptext_eve.cpu().detach().numpy()

    print('Real ptext:\n{}\n\nptext bob:\n{}\n\nptext eve:\n{}'.format(ptext, predict_ptext_bob, predict_ptext_eve))


if __name__ == '__main__':

    validate()






