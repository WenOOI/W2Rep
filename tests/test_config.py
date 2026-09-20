from pathlib import Path

from w2rep.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_final_config_and_inherited_ablation_load():
    final, final_hash = load_config(ROOT / "configs/pretrain/w2rep_vitb16.yaml")
    ablation, ablation_hash = load_config(
        ROOT / "configs/ablations/no_signed_offset.yaml"
    )
    assert final.objective.signed_offset is True
    assert ablation.objective.signed_offset is False
    assert ablation.model.architecture == "vitb16"
    assert final_hash != ablation_hash

