import unittest

import level_aware_recommendation as level_aware


def fixed_name_score(score: float):
    def _score(
        fhir_terms: set[str],
        column: dict[str, object],
        fhir_path_tail: str,
    ) -> float:
        return score

    return _score


def no_type_score(
    fhir_type: str,
    data_type: str,
    name_terms: set[str],
) -> float:
    return 0.0


def no_distance_score(distance: int, max_distance: int) -> float:
    return 0.0


class LevelAwareRecommendationTests(unittest.TestCase):
    def test_seed_order_and_fk_distance_drive_table_priority(self) -> None:
        priorities = level_aware.build_table_priorities(
            ["HIGH", "LOW"],
            {"HIGH": {"HIGH_CHILD"}, "HIGH_CHILD": {"HIGH"}},
            max_distance=2,
        )

        self.assertGreater(priorities["HIGH"].priority, priorities["LOW"].priority)
        self.assertGreater(
            priorities["HIGH"].priority,
            priorities["HIGH_CHILD"].priority,
        )
        self.assertEqual(priorities["HIGH"].distance, 0)
        self.assertEqual(priorities["HIGH_CHILD"].distance, 1)

    def test_weak_table_candidate_is_suppressed_when_field_score_is_not_strong(
        self,
    ) -> None:
        columns = [
            {
                "table": "HIGH",
                "column": "VALUE",
                "data_type": "",
                "distance": 0,
                "name_terms": set(),
                "table_priority": 1.0,
                "table_level": 0,
            },
            {
                "table": "WEAK",
                "column": "VALUE",
                "data_type": "",
                "distance": 0,
                "name_terms": set(),
                "table_priority": 0.2,
                "table_level": 5,
            },
        ]

        matches = level_aware.rank_candidates_for_row(
            fhir_terms={"value"},
            fhir_path="Patient.value",
            fhir_type="string",
            desc_row=[0.0, 0.0],
            columns=columns,
            weights={"name": 1.0, "desc": 0.0, "type": 0.0, "dist": 0.0},
            max_distance=2,
            top_k=0,
            min_score=0.05,
            name_similarity_fn=fixed_name_score(0.6),
            type_compat_fn=no_type_score,
            distance_score_fn=no_distance_score,
        )

        self.assertEqual([match[2]["table"] for match in matches], ["HIGH"])

    def test_same_field_score_is_ordered_by_table_priority(self) -> None:
        columns = [
            {
                "table": "LOW",
                "column": "VALUE",
                "data_type": "",
                "distance": 0,
                "name_terms": set(),
                "table_priority": 0.5,
                "table_level": 1,
            },
            {
                "table": "HIGH",
                "column": "VALUE",
                "data_type": "",
                "distance": 0,
                "name_terms": set(),
                "table_priority": 1.0,
                "table_level": 0,
            },
        ]

        matches = level_aware.rank_candidates_for_row(
            fhir_terms={"value"},
            fhir_path="Patient.value",
            fhir_type="string",
            desc_row=[0.0, 0.0],
            columns=columns,
            weights={"name": 1.0, "desc": 0.0, "type": 0.0, "dist": 0.0},
            max_distance=2,
            top_k=0,
            min_score=0.05,
            name_similarity_fn=fixed_name_score(0.8),
            type_compat_fn=no_type_score,
            distance_score_fn=no_distance_score,
        )

        self.assertEqual([match[2]["table"] for match in matches], ["HIGH", "LOW"])

    def test_enriched_column_text_includes_table_and_column_context(self) -> None:
        summaries = {
            "PERSON": level_aware.TableSummary(
                description="Person demographics and identifiers",
                column_descriptions={"BIRTH_DT_TM": "Date and time the person was born"},
            )
        }
        column = {
            "table": "PERSON",
            "column": "BIRTH_DT_TM",
            "definition": "Birth timestamp",
            "data_type": "DATE",
            "name_terms": {"birth", "date", "time"},
            "table_terms": {"person"},
        }

        text = level_aware.build_enriched_column_text(column, summaries)

        self.assertIn("Person demographics and identifiers", text)
        self.assertIn("Date and time the person was born", text)
        self.assertIn("BIRTH_DT_TM", text)
        self.assertIn("DATE", text)
        self.assertIn("birth", text)


if __name__ == "__main__":
    unittest.main()
