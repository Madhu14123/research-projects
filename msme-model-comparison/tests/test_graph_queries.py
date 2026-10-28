import json
import unittest
from pathlib import Path

from graph_viewer import execute_nl_query, parse_nl_query


class GraphQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph_path = Path("outputs/msme_knowledge_graph_extensive.json")
        with open(cls.graph_path, "r", encoding="utf-8") as fh:
            cls.graph = json.load(fh)

    def test_parse_women_query(self):
        parsed = parse_nl_query("show women entrepreneurs who need support")
        self.assertEqual(parsed["intent"], "women")

    def test_parse_cluster_query(self):
        parsed = parse_nl_query("find similar clusters and related industries")
        self.assertEqual(parsed["intent"], "cluster")

    def test_execute_women_query_returns_matches(self):
        result = execute_nl_query("show women entrepreneurs who need support", self.graph)
        self.assertGreater(result["match_count"], 0)
        self.assertIn("women", result["summary"].lower())
        self.assertIn("why", result["explanation"].lower())
        self.assertTrue(result["matches"][0]["why_it_matched"])

    def test_execute_supply_chain_query_returns_matches(self):
        result = execute_nl_query("find supply chain linkages between industries", self.graph)
        self.assertGreater(result["match_count"], 0)
        self.assertIn("supply", result["summary"].lower())
        self.assertIn("evidence", result["explanation"].lower())


if __name__ == "__main__":
    unittest.main()
