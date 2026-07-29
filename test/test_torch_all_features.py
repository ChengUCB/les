"""TorchScript regression test exercising all long-range features together.

Follows the style of test_torch.py / test_torch_dipole.py: build a model, run it
eagerly, script it with torch.jit.script, save/reload the scripted module, run it,
and check the scripted outputs match the eager outputs.

Features exercised simultaneously:
  - monopoles (latent_charges)
  - dipoles (latent_dipoles)
  - quadrupoles (latent_quads)
  - atomic alpha (use_atomic_alpha + latent_alphas + atomic_numbers)
  - atomic kappa (latent_kappas)

Also regression-checks backward compatibility: a model whose serialized state is
missing the newer use_* flags (as in checkpoints saved before they existed) must
still load, run, and script with identical results (see Les.__setstate__).
"""
import io
import tempfile

import torch

from les import Les


def _build_inputs(seed: int = 0):
    torch.manual_seed(seed)
    n = 10
    r = torch.rand(n, 3) * 10
    r.requires_grad_(True)

    q = torch.rand(n) * 2 - 1
    q = q - torch.mean(q)                     # monopoles (charge-neutral)
    u = torch.rand(n, 3) * 2
    u = u - torch.mean(u, 0)                  # dipoles
    quad = torch.rand(n, 3, 3)
    quad = 0.5 * (quad + quad.transpose(1, 2))  # symmetric quadrupoles
    kappa = torch.ones(n) * 0.5               # atomic kappa (hardness)
    alpha = torch.ones(n) * 0.5               # (isotropic) latent alpha
    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1, 8, 1, 1, 8], dtype=torch.int64)

    box = torch.tensor([[10.0, 0.0, 0.0],
                        [0.0, 10.0, 0.0],
                        [0.0, 0.0, 10.0]]).unsqueeze(0)
    return dict(positions=r, cell=box, latent_charges=q, latent_dipoles=u,
                latent_quads=quad, latent_kappas=kappa, latent_alphas=alpha,
                atomic_numbers=atomic_numbers, batch=None, compute_bec=False)


def _run(model, inputs):
    return model(
        positions=inputs['positions'],
        cell=inputs['cell'],
        latent_charges=inputs['latent_charges'],
        latent_dipoles=inputs['latent_dipoles'],
        latent_quads=inputs['latent_quads'],
        latent_kappas=inputs['latent_kappas'],
        latent_alphas=inputs['latent_alphas'],
        atomic_numbers=inputs['atomic_numbers'],
        batch=inputs['batch'],
        compute_bec=inputs['compute_bec'],
    )


def _assert_scripted_matches_eager(model, inputs, label):
    """Script, save/reload, run, and compare against the eager result."""
    eager = _run(model, inputs)

    scripted = torch.jit.script(model)
    with tempfile.NamedTemporaryFile() as tmp:      # scripted module is saveable
        torch.jit.save(scripted, tmp.name)
        scripted = torch.jit.load(tmp.name)

    scripted_out = _run(scripted, inputs)           # scripted module runs
    for _ in range(3):                              # runs repeatedly (stateless)
        scripted_out = _run(scripted, inputs)

    assert eager.keys() == scripted_out.keys(), f"[{label}] output keys differ"
    checked = []
    for k, v in eager.items():
        if v is None:
            assert scripted_out[k] is None, f"[{label}] key {k}: eager None, scripted not"
            continue
        assert scripted_out[k] is not None, f"[{label}] key {k}: scripted None, eager not"
        assert torch.allclose(v, scripted_out[k], atol=1e-6), (
            f"[{label}] key {k} mismatch:\n eager={v.flatten()[:5]}\n "
            f"scripted={scripted_out[k].flatten()[:5]}")
        checked.append(k)
    # energy and the enabled electrostatic outputs must be present and checked
    for required in ('E_lr', 'latent_charges', 'latent_dipoles',
                     'latent_quads', 'latent_alphas'):
        assert required in checked, f"[{label}] expected output {required} not verified"
    print(f"[{label}] scripted == eager for keys: {checked}")


def test_all_features_scriptable_and_consistent():
    inputs = _build_inputs()
    model = Les(les_arguments={
        'use_atomwise': False,
        'use_atomic_alpha': True,      # exercises AtomicAlpha baseline addition
        'remove_self_interaction': True,
    })
    _assert_scripted_matches_eager(model, inputs, "all-features")


def test_backward_compat_old_checkpoint_scriptable_and_consistent():
    """A checkpoint saved before the use_* flags existed must still load,
    run, and script with identical results after __setstate__ migration."""
    inputs = _build_inputs()
    model = Les(les_arguments={'use_atomic_alpha': True})

    # Simulate an old serialized model by stripping the newer flags, matching
    # what a pre-flag pickle's __dict__ would contain, then reload.
    buf = io.BytesIO()
    torch.save(model, buf); buf.seek(0)
    obj = torch.load(buf, weights_only=False)
    for flag in ('use_atomwise', 'use_fixed_atomic_charges',
                 'use_atomic_alpha', 'use_epsilon_r_scaling'):
        obj.__dict__.pop(flag, None)
    buf2 = io.BytesIO()
    torch.save(obj, buf2); buf2.seek(0)
    migrated = torch.load(buf2, weights_only=False)

    for flag in ('use_atomwise', 'use_fixed_atomic_charges',
                 'use_atomic_alpha', 'use_epsilon_r_scaling'):
        assert hasattr(migrated, flag), f"__setstate__ did not restore {flag}"

    # Old checkpoints predate use_atomic_alpha, so it is restored to False;
    # drop the atomic-alpha baseline path for an apples-to-apples comparison.
    compat_inputs = dict(inputs)
    compat_inputs['atomic_numbers'] = None
    _assert_scripted_matches_eager(migrated, compat_inputs, "backward-compat")


if __name__ == "__main__":
    test_all_features_scriptable_and_consistent()
    test_backward_compat_old_checkpoint_scriptable_and_consistent()
    print("\nAll TorchScript regression checks passed.")
