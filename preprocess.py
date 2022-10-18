import esm
import pathlib
import numpy as np
import torch

# sequence embedding from pre-trained ESM-1b
def get_embedding_esm(input_fasta):
    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_model.eval()

    if torch.cuda.is_available():
        esm_model = esm_model.cuda()
        print("Transferred model to GPU")
    seq_dataset = esm.FastaBatchedDataset.from_file(input_fasta)
    batches = seq_dataset.get_batch_indices(2, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(seq_dataset, collate_fn = alphabet.get_batch_converter(), batch_sampler=batches)
    print(f"Read .fasta file with {len(seq_dataset)} sequences")
    output_dir = pathlib.Path('./data/raw/')
    output_dir.mkdir(parents=True, exist_ok=True)
    include = ["mean", "per_tok","contacts"]
    return_contacts = "contacts" in include

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = esm_model(toks, repr_layers=[33], return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                output_file = (output_dir / f"{label}.pt")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                if "per_tok" in include:
                        result["representations"] = {
                            layer: t[i, 1 : len(strs[i]) + 1].clone()
                            for layer, t in representations.items()
                        }
                if "mean" in include:
                        result["mean_representations"] = {
                            layer: t[i, 1 : len(strs[i]) + 1].mean(0).clone()
                            for layer, t in representations.items()
                        }
                if "bos" in include:
                        result["bos_representations"] = {
                            layer: t[i, 0].clone() for layer, t in representations.items()
                            }
                if return_contacts:
                        result["contacts"] = contacts[i, :len(strs[i]), :len(strs[i])].clone()
                torch.save(result, output_file)
                break

# get contact maps
def get_contact():
    max_length = 65
    contact = np.zeros((max_length,max_length))

    path_raw= "./data/raw/"
    contact_path = "./data/contact_65/"
    esm_path = "./data/embedding_65/"
    sample_list = os.listdir(path_raw)
    for pr in sample_list:
        path = path_raw + pr
        model = torch.load(path)
        model_pth = model["contacts"]
        esm = model['representations'][33]
        esm_size = esm.shape[0]
        if esm_size < max_length:
            zero = np.zeros((max_length - esm_size, 1280))
            zero1 = torch.from_numpy(zero)
            esm = torch.cat((esm, zero1), 0)
        path = esm_path + pr
        model_esm = {'feature':esm,'size':esm_size}
        torch.save(model_esm,path)

        size = model_pth.shape
        seq_len = size[0]
        if seq_len < max_length:
            for i in range(seq_len):
                for j in range(seq_len):
                    contact[i][j]=model_pth[i][j]
        else:
            for i in range(max_length):
                for j in range(max_length):
                    contact[i][j]=model_pth[i][j]
        pr_name = pr[:-2]
        path = contact_path + pr_name + "npy" 
        np.save(path, contact)

if __name__ == '__main__':
    input_fasta = './data/pepvae.fasta'
    get_embedding_esm(input_fasta)
    get_contact()
