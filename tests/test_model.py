import numpy as np
import pytest
from pydantic import ValidationError

from superconductor import Filling, HubbardModel, SquareLattice
from superconductor.model import load_model


def make(**kw):
    base = dict(lattice=SquareLattice(lx=2, ly=2), U=4.0, filling=Filling(n_up=2, n_dn=2))
    return HubbardModel(**{**base, **kw})


def test_json_roundtrip_preserves_model_and_fingerprint(tmp_path):
    m = make(t_prime=-0.3, name="x")
    path = tmp_path / "m.json"
    path.write_text(m.to_json())
    loaded = load_model(path)
    assert loaded == m
    assert loaded.fingerprint() == m.fingerprint()


def test_fingerprint_tracks_physics_not_label():
    m = make()
    assert make(name="something else").fingerprint() == m.fingerprint()
    assert make(U=4.0000001).fingerprint() != m.fingerprint()
    assert make(filling=Filling(n_up=2, n_dn=1)).fingerprint() != m.fingerprint()


def test_rejects_impossible_filling_and_unknown_fields():
    with pytest.raises(ValidationError):
        make(filling=Filling(n_up=5, n_dn=0))
    with pytest.raises(ValidationError):
        HubbardModel.model_validate({**make().model_dump(), "Jz": 1.0})


def test_doping():
    m = HubbardModel(lattice=SquareLattice(lx=4, ly=2), filling=Filling(n_up=3, n_dn=3))
    assert m.density == pytest.approx(0.75)
    assert m.hole_doping == pytest.approx(0.25)


@pytest.mark.parametrize("lx,ly", [(2, 2), (4, 2), (3, 3), (4, 4), (6, 1)])
def test_periodic_hopping_matches_tight_binding_dispersion(lx, ly):
    t, tp = 1.0, -0.3
    m = HubbardModel(
        lattice=SquareLattice(lx=lx, ly=ly), t=t, t_prime=tp, filling=Filling(n_up=0, n_dn=0)
    )
    h = m.hopping_matrix()
    assert np.allclose(h, h.T)
    k = m.lattice.allowed_momenta()
    eps = -2 * t * (np.cos(k[:, 0]) * (lx > 1) + np.cos(k[:, 1]) * (ly > 1))
    if lx > 1 and ly > 1:
        eps += -4 * tp * np.cos(k[:, 0]) * np.cos(k[:, 1])
    assert np.allclose(np.sort(np.linalg.eigvalsh(h)), np.sort(eps))


def test_open_chain_has_no_wraparound_bond():
    m = HubbardModel(
        lattice=SquareLattice(lx=3, ly=1, boundary_x="open"), filling=Filling(n_up=0, n_dn=0)
    )
    h = m.hopping_matrix()
    assert h[0, 2] == 0 and h[0, 1] == -1.0
