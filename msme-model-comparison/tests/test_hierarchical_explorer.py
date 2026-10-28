import json
import unittest

import pandas as pd

from graph_viewer import build_drilldown_subgraph, derive_question_taxonomy, validate_schema


class HierarchicalExplorerTests(unittest.TestCase):
    def test_question_taxonomy_includes_support_and_risk_questions(self):
        df = pd.DataFrame(
            {
                "DISTRICT_NAME": ["A", "A", "B"],
                "CATEGOERY_NAME": ["small", "small", "large"],
                "investment": [100.0, 200.0, 500.0],
                "employment": [10, 20, 50],
                "promoter_women": ["Yes", "No", "Yes"],
                "PCBCATEGORY": ["GREEN", "RED", "ORANGE"],
            }
        )

        taxonomy = derive_question_taxonomy(df)
        self.assertGreater(len(taxonomy["question_types"]), 0)
        self.assertTrue(any(item["area"] == "support" for item in taxonomy["question_types"]))
        self.assertTrue(any(item["area"] == "risk" for item in taxonomy["question_types"]))

    def test_build_drilldown_subgraph_returns_root_and_neighbors(self):
        nodes = [
            {"id": "msme-1", "type": "MSME", "label": "Unit A", "properties": {}},
            {"id": "district-1", "type": "District", "label": "District A", "properties": {}},
            {"id": "mandal-1", "type": "Mandal", "label": "Mandal A", "properties": {}},
            {"id": "risk-1", "type": "RiskBucket", "label": "High", "properties": {}},
        ]
        edges = [
            {"source": "msme-1", "target": "district-1", "relation": "LOCATED_IN"},
            {"source": "msme-1", "target": "mandal-1", "relation": "OPERATES_IN"},
            {"source": "msme-1", "target": "risk-1", "relation": "HAS_RISK_BUCKET"},
        ]

        result = build_drilldown_subgraph(nodes, edges, "msme-1", depth=2)
        self.assertIn("msme-1", result["node_ids"])
        self.assertIn("district-1", result["node_ids"])
        self.assertIn("risk-1", result["node_ids"])
        self.assertEqual(result["depth"], 2)

    def test_schema_validation_accepts_current_schema(self):
        with open("knowledge_graph_schema_extensive.json", "r", encoding="utf-8") as fh:
            schema = json.load(fh)

        report = validate_schema(schema)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
