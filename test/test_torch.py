import sys
sys.path.append('../')
from les import Les
import logging, traceback
import torch
import tempfile

#load the model
les = Les(les_arguments={})



# set the same random seed for reproducibility
torch.manual_seed(0)
r = torch.rand(10, 3) * 10  # Random positions in a 10x10x10 box
desc = torch.rand(10, 3) * 10  # Random descriptors
r.requires_grad_(requires_grad=True)
q = torch.rand(10) * 2 - 1 # Random charges

box = torch.tensor([10.0, 10.0, 10.0])  # Box dimensions
box_full = torch.tensor([
    [10.0, 0,0],
    [0,10.0, 0], 
    [0,0,10.0]])  # Box dimensions

result = les(desc=desc,
    positions=r,
    cell=box_full.unsqueeze(0),
    batch=None,
    compute_bec=True,)




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
    script_result = scripted_model(desc=desc,
                                    positions=r,
                                    cell=box_full.unsqueeze(0),
                                    batch=None,
                                    compute_bec=True,)
    with tempfile.NamedTemporaryFile() as tmp:
        torch.jit.save(scripted_model, tmp.name)
    print("Model scripted and saved successfully.")
except Exception as e:
    logging.error(f"Error scripting the model: {e}")
    logging.error(traceback.format_exc())

#check the results consistency
if result.keys() != script_result.keys():
    print("Keys do not match")
else:
    for k in result.keys():
        print(f"{k}: {torch.allclose(result[k], script_result[k])}")