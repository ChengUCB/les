from les import Les
import logging, traceback
import torch
import tempfile

#load the model
les = Les(les_arguments={
    'use_atomwise': False, # default: False
    'remove_self_interaction': True, # default: True
    'dl': 1.5,
})


# set the same random seed for reproducibility
torch.manual_seed(0)
r = torch.rand(10, 3) * 10  # Random positions in a 10x10x10 box
r.requires_grad_(requires_grad=True)
q = torch.rand(10) * 2 - 1 # Random charges
q = q - torch.mean(q)

# dipole, kappa, alpha
u = torch.rand(10, 3) * 2
u -= torch.mean(u, 0)
kappa = torch.ones(10) * 0.5
alpha = torch.ones(10) * 0.5

box_size = 10.0 # test the case with periodicity
# box_size = 0.0 # test the case with no periodicity

box_full = torch.tensor([
    [box_size, 0,0],
    [0,box_size, 0], 
    [0,0,box_size]])  # Box dimensions

result = les(
    positions=r,
    cell=box_full.unsqueeze(0),
    desc = None,
    latent_charges=q,
    latent_dipoles=u,
    latent_kappas=kappa,
    latent_alphas=alpha,
    batch=None,
    compute_bec=False,)
# torch.save(les, "./saved_modules/les.pt")
# print(result)

# scripting save all modules
for name, module in les.named_modules():
    try:
        scripted_module = torch.jit.script(module)
        # print(f"Scripted {name}")
        with tempfile.NamedTemporaryFile() as tmp:
            torch.jit.save(scripted_module, tmp.name)
        # print(f"Saved: {name}")
    except Exception as e:
        print(f"Save failed: {name}, Error: {e}")

#scripting and save the whole model
try:
    scripted_model = torch.jit.script(les)
    with tempfile.NamedTemporaryFile() as tmp:
        torch.jit.save(scripted_model, tmp.name)
    print("Model scripted and saved successfully.")
except Exception as e:
    logging.error(f"Error scripting or saving the model: {e}")
    logging.error(traceback.format_exc())

#multiple inferences test
try:
    for i in range(3):
        script_result = scripted_model(positions=r,
                                       cell=box_full.unsqueeze(0),
                                       desc = None,
                                       latent_charges=q,
                                       latent_dipoles=u,
                                       latent_kappas=kappa,
                                       latent_alphas=alpha,
                                       batch=None,
                                        compute_bec=False,)  
except Exception as e:
    logging.error(f"Error: {e}")
    logging.error(traceback.format_exc())

#check the results consistency
if result.keys() != script_result.keys():
    print("Keys do not match")
else:
    for k in result.keys():
        if result[k] is not None:
            if torch.allclose(result[k], script_result[k], atol=1e-6):
                print(f'key: {k} \n Torchscript result is identical to original result. \n')
            else:
                print(f'''Key: {k} \n Torchscript result is different from original result. \n
                    Original result: {result[k][:3]} \n Torchscript result: {script_result[k][:3]}''')