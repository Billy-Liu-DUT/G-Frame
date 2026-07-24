import unittest

from g_frame.clients import ScriptedChatClient
from g_frame.data import SFTDatasetBuilder
from g_frame.prompts import PromptCatalog
from g_frame.team_game import TeamGame


class TeamGameTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejection_triggers_rectification_and_second_review(self):
        client = ScriptedChatClient(
            {
                "teacher": '{"question": "What controls the color shift?"}',
                "student": '{"reasoning": "initial", "answer": "too broad"}',
                "reviewer": [
                    '{"approved": false, "feedback": "Add the electronic explanation."}',
                    '{"approved": true, "feedback": "Grounded now."}',
                ],
                "rectifier": '{"reasoning": "gap decreases", "answer": "Conjugation often lowers the electronic gap."}',
                "judger": '{"final_answer": "Conjugation often lowers the electronic gap, subject to molecular geometry.", "approved": true, "feedback": "Final answer is grounded."}',
            }
        )
        game = TeamGame(client, PromptCatalog(), model="scripted", max_revisions=1)
        record = await game.run("task-1", "source-1", "Conjugation affects transition energy.")

        self.assertTrue(record.approved)
        self.assertIn("geometry", record.final_answer)
        self.assertEqual(
            [item["role"] for item in record.agent_trace],
            ["Teacher", "Student", "Reviewer", "Rectifier", "Reviewer", "Judger"],
        )
        self.assertIn("<think>gap decreases</think>", record.to_chat_messages()[-1]["content"])

    async def test_empty_teacher_question_is_rejected(self):
        client = ScriptedChatClient({"teacher": '{"question": ""}'})
        game = TeamGame(client, PromptCatalog(), model="scripted")
        with self.assertRaisesRegex(ValueError, "Teacher"):
            await game.run("task-1", "source-1", "A non-empty source.")

    async def test_json_fenced_agent_payload_is_parsed(self):
        client = ScriptedChatClient(
            {
                "teacher": '```json\n{"question": "What is the trend?"}\n```',
                "student": '{"reasoning": "initial", "answer": "A trend."}',
                "reviewer": '{"approved": true, "feedback": "Grounded."}',
                "judger": '{"final_answer": "A trend.", "approved": true, "feedback": "Grounded."}',
            }
        )
        record = await TeamGame(client, PromptCatalog(), model="scripted").run("task-3", "source-3", "A source.")
        self.assertEqual(record.question, "What is the trend?")

    async def test_boolean_approved_alias_is_honored_by_reviewer_and_judger(self):
        client = ScriptedChatClient(
            {
                "teacher": '{"question": "What is the trend?"}',
                "student": '{"reasoning": "initial", "answer": "A trend."}',
                "reviewer": '{"boolean_approved": true, "feedback": "Grounded."}',
                "judger": '{"final_answer": "A trend.", "boolean_approved": true, "feedback": "Grounded."}',
            }
        )
        record = await TeamGame(client, PromptCatalog(), model="scripted").run("task-4", "source-4", "A source.")
        self.assertTrue(record.approved)

    async def test_boolean_approved_judger_rejection_is_honored(self):
        client = ScriptedChatClient(
            {
                "teacher": '{"question": "What is the trend?"}',
                "student": '{"reasoning": "initial", "answer": "A trend."}',
                "reviewer": '{"approved": true, "feedback": "Candidate is grounded."}',
                "judger": '{"final_answer": "Unsupported expansion.", "boolean_approved": false, "feedback": "Not grounded."}',
            }
        )
        record = await TeamGame(client, PromptCatalog(), model="scripted").run("task-5", "source-5", "A source.")
        self.assertFalse(record.approved)

    async def test_unapproved_judger_output_is_excluded_from_sft_rows(self):
        client = ScriptedChatClient(
            {
                "teacher": '{"question": "What is the trend?"}',
                "student": '{"reasoning": "initial", "answer": "A trend."}',
                "reviewer": '{"approved": true, "feedback": "Candidate is grounded."}',
                "judger": '{"final_answer": "Unsupported expansion.", "approved": false, "feedback": "Not grounded."}',
            }
        )
        record = await TeamGame(client, PromptCatalog(), model="scripted").run("task-2", "source-2", "A source.")
        self.assertFalse(record.approved)
        self.assertEqual(SFTDatasetBuilder.build_rows([record]), [])
