import unittest

from g_frame.augmentation import AUGMENTATION_STYLES, SixStyleAugmenter
from g_frame.clients import ScriptedChatClient
from g_frame.prompts import PromptCatalog


class AugmentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_six_styles_preserve_lineage_and_use_editable_prompts(self):
        client = ScriptedChatClient(
            {
                f"augment_{style}": f'{{"augmented_text": "{style} grounded rewrite"}}'
                for style in AUGMENTATION_STYLES
            }
        )
        augmenter = SixStyleAugmenter(client, PromptCatalog(), model="scripted")
        rows = await augmenter.augment("source-7", "A grounded chemistry passage.")
        self.assertEqual([row.style for row in rows], list(AUGMENTATION_STYLES))
        self.assertTrue(all(row.source_id == "source-7" for row in rows))
        self.assertEqual(len(client.requests), 6)

    async def test_unknown_style_is_rejected(self):
        augmenter = SixStyleAugmenter(ScriptedChatClient({}), PromptCatalog(), model="scripted")
        with self.assertRaisesRegex(ValueError, "unknown"):
            await augmenter.augment("source-7", "A passage.", styles=("invented",))
