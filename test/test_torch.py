from les import Les
import logging, traceback
import torch
import tempfile

#load the model
les = Les(les_arguments={'use_atomwise': True}) # desc is used below


# set the same random seed for reproducibility
torch.manual_seed(0)
r = torch.rand(10, 3) * 10  # Random positions in a 10x10x10 box
r.requires_grad_(requires_grad=True)
q = torch.rand(10) * 2 - 1 # Random charges
box_full = torch.tensor([
    [10.0, 0,0],
    [0,10.0, 0], 
    [0,0,10.0]])  # Box dimensions

result = les(desc=r,
    positions=r,
    cell=box_full.unsqueeze(0),
    batch=None,
    compute_bec=False,)
# torch.save(les, "./saved_modules/les.pt")

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
        script_result = scripted_model(desc=r,
                                        positions=r,
                                        cell=box_full.unsqueeze(0),
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
        if result[k] is not None and torch.allclose(result[k], script_result[k]):
            print(f"Key: {k} \n Torchscript result is identical to original result.")


les_no_atomwise = Les(les_arguments={'use_atomwise': False})
torch.jit.script(les_no_atomwise)
result = les(latent_charges=q,
                positions=r,
                cell=box_full.unsqueeze(0),
                batch=None,
                compute_bec=False,)