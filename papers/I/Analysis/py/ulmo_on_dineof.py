""" Run Ulmo on the DINEOF images for LL"""

# imports
import os
import glob
import numpy as np

import torch
import h5py

import pickle
import pandas

from tqdm.auto import tqdm

from ulmo import io as ulmo_io
from ulmo.preproc import utils as pp_utils
from ulmo.plotting import plotting
from ulmo.models import autoencoders, ConditionalFlow
from ulmo import ood

from IPython import embed

def main():

    # Load data
    tbl_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'Tables')
    tbl_file = os.path.join(tbl_path, 'Enki_LLC_DINOEF.parquet')
    dineof_tbl = ulmo_io.load_main_table(tbl_file)

    preproc_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'PreProc')
    preproc_file = os.path.join(preproc_path, 'Enki_LLC_DINEOF_preproc.h5')
    f_data = h5py.File(preproc_file, 'r')
    
    # Load model
    model_path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Ulmo')
    print("Loading model in {}".format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcae = autoencoders.DCAE.from_file(os.path.join(model_path, 'autoencoder.pt'),
                                       image_shape=(1, 64, 64),
                                       latent_dim=512)
    flow = ConditionalFlow(
        dim=512,
        context_dim=None,
        transform_type='autoregressive',
        n_layers=10,
        hidden_units=256,
        n_blocks=2,
        dropout=0.2,
        use_batch_norm=False,
        tails='linear',
        tail_bound=10,
        n_bins=5,
        min_bin_height=1e-3,
        min_bin_width=1e-3,
        min_derivative=1e-3,
        unconditional_transform=False,
        encoder=None)
    flow.load_state_dict(torch.load(os.path.join(model_path, 'flow.pt'), map_location=device))
    pae = ood.ProbabilisticAutoencoder(dcae, flow, 'tmp/', device=device, skip_mkdir=True)
    print("Model loaded!")

    pae.autoencoder.eval()
    pae.flow.eval()

    # Scaler
    scaler_path = glob.glob(os.path.join(model_path, '*scaler.pkl'))[0]
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    LLs = []
    for ss in range(f_data['valid'].shape[0]):
        pp_field = f_data['valid'][ss, ...]
        # Latent
        pp_field.resize(1, 1, 64, 64)
        dset = torch.utils.data.TensorDataset(torch.from_numpy(pp_field).float())
        loader = torch.utils.data.DataLoader(
            dset, batch_size=1, shuffle=False,
            drop_last=False, num_workers=16)
        with torch.no_grad():
            latents = [pae.autoencoder.encode(data[0].to(device)).detach().cpu().numpy()
                    for data in tqdm(loader, total=len(loader), unit='batch', desc='Computing latents')]
        print("Latents generated!")
        latents = scaler.transform(np.concatenate(latents))

        # LL
        dset = torch.utils.data.TensorDataset(torch.from_numpy(latents).float())
        loader = torch.utils.data.DataLoader(
            dset, batch_size=1024, shuffle=False,
            drop_last=False, num_workers=16)

        with torch.no_grad():
            log_prob = [pae.flow.log_prob(data[0].to(pae.device)).detach().cpu().numpy()
                            for data in tqdm(loader, total=len(loader), unit='batch', desc='Computing log probs')]
        print("Log probabilities generated!")

        print(f"The LL for the field {ss} is: {float(log_prob[0])}")

        # LLs
        LLs.append(float(log_prob[0]))

    # Stats
    print(f"Mean LL: {np.mean(LLs)}")
    print(f"Median LL: {np.median(LLs)}")
    print(f"Min LL: {np.min(LLs)}")
    print(f"Max LL: {np.max(LLs)}")

    embed(header='107 of main')
    df = pandas.DataFrame({'LL': LLs})
    df.to_csv('LLs.csv')

    from matplotlib import pyplot as plt
    ax = plt.gca()
    ax.hist(LLs, bins=20)
    plotting.set_fontsize(ax, 15)
    plt.show()

if __name__ == '__main__':
    main()