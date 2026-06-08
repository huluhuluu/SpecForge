import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from specforge.utils import generate_draft_model_config


class TestDraftConfigGeneration(unittest.TestCase):
    def _generate_config(self, draft_attention_type="target"):
        target_config = SimpleNamespace(
            model_type="llama",
            vocab_size=128256,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            bos_token_id=128000,
            eos_token_id=128001,
            torch_dtype=torch.float16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = f"{tmpdir}/template.json"
            with open(template_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "architectures": ["LlamaForCausalLM"],
                        "draft_vocab_size": 32000,
                    },
                    f,
                )

            with patch(
                "specforge.utils.AutoConfig.from_pretrained",
                return_value=target_config,
            ):
                return generate_draft_model_config(
                    "dummy-target",
                    template_config_path=template_path,
                    draft_attention_type=draft_attention_type,
                )

    def test_generate_draft_model_config_preserves_target_attention_by_default(self):
        draft_config = self._generate_config()

        self.assertEqual(draft_config["num_attention_heads"], 32)
        self.assertEqual(draft_config["num_key_value_heads"], 8)

    def test_generate_draft_model_config_sets_mha_when_requested(self):
        draft_config = self._generate_config(draft_attention_type="mha")

        self.assertEqual(
            draft_config["num_key_value_heads"], draft_config["num_attention_heads"]
        )
        self.assertEqual(draft_config["num_key_value_heads"], 32)


if __name__ == "__main__":
    unittest.main()
