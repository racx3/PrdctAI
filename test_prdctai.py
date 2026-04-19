import unittest

from prdctai import demo, exploit_human_random, generate_human_like_sequence


class PredictableRandomTests(unittest.TestCase):
    def test_generate_human_like_sequence_shape(self):
        sequence = generate_human_like_sequence(10, switch_bias=0.6, seed=123)
        self.assertEqual(len(sequence), 10)
        self.assertTrue(all(choice in (0, 1) for choice in sequence))

    def test_predictor_exploits_alternating_pattern(self):
        sequence = [i % 2 for i in range(200)]
        _, accuracy = exploit_human_random(sequence)
        self.assertGreater(accuracy, 0.95)

    def test_predictor_beats_chance_on_human_like_bias(self):
        sequence = generate_human_like_sequence(400, switch_bias=0.72, seed=42)
        _, accuracy = exploit_human_random(sequence)
        self.assertGreater(accuracy, 0.55)

    def test_demo_reports_accuracy(self):
        output = demo(length=50, switch_bias=0.7, seed=42)
        self.assertIn("accuracy", output)
        self.assertIn("switch_bias=0.7", output)


if __name__ == "__main__":
    unittest.main()
