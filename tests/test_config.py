import json
from pathlib import Path

from inkling_engine.config import InklingConfig

CFG = json.loads((Path(__file__).parent / "data" / "config.json").read_text())


def test_visits_layout():
    c = InklingConfig.from_dict(CFG)
    v = c.visits
    assert c.num_visits == len(v) == 48
    assert [x.layer for x in v[:6]] == list(range(6))
    assert [x.layer for x in v[6:18]] == list(range(6, 18))
    assert [x.pass_idx for x in v[18:30]] == [1] * 12
    assert [x.layer for x in v[42:]] == list(range(18, 24))
    glob = [x.layer for x in v if not x.is_local]
    assert glob == [5, 11, 17, 11, 17, 11, 17, 23]
    assert c.num_global_visits == 8 and c.num_local_visits == 40
    assert c.q_dim == 3072 and c.kv_dim == 1024 and c.rel_dim == 384


def test_derived_constants():
    c = InklingConfig.from_dict(CFG)
    assert c.top_k == 8 and c.n_experts == 128 and c.moe_inter == 1152
    assert abs(c.conv_out_scale - 5 ** -0.5) < 1e-12
    assert abs(c.dyn_conv_out_scale - 1 / 3) < 1e-12
    assert c.mtp["target_layer_ids"] == [5, 23, 41, 48]
