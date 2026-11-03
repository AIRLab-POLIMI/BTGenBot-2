import tempfile
import unittest
from pathlib import Path

from model.bt_validator import BTValidationError, BehaviorTreeValidator


VOCABULARY_PATH = Path(__file__).parents[1] / "model" / "allowed_primitives.yaml"


def tree(body: str, *, root_attributes: str = 'BTCPP_format="4"') -> str:
    return (
        f"<root {root_attributes}>"
        '<BehaviorTree ID="MainTree">'
        f"{body}"
        "</BehaviorTree>"
        "</root>"
    )


class BehaviorTreeValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = BehaviorTreeValidator.from_yaml(VOCABULARY_PATH)

    def assert_invalid(self, xml_text: str, message: str) -> None:
        with self.assertRaisesRegex(BTValidationError, message):
            self.validator.validate(xml_text)

    def test_accepts_compact_action_nodes(self) -> None:
        xml_text = tree(
            "<Sequence>"
            '<MoveTo location="Warehouse Left"/>'
            '<MoveTo name="forklift_goal" location="Warehouse Forklift"/>'
            "</Sequence>"
        )

        root = self.validator.validate(xml_text)

        self.assertEqual(root.tag, "root")

    def test_accepts_explicit_action_nodes(self) -> None:
        xml_text = tree('<Action ID="MoveTo" location="Warehouse"/>')

        self.validator.validate(xml_text)

    def test_accepts_navigation_and_recovery_primitives(self) -> None:
        xml_text = tree(
            '<RecoveryNode number_of_retries="3">'
            "<PipelineSequence>"
            '<RateController hz="1.0">'
            '<ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>'
            "</RateController>"
            '<FollowPath path="{path}" controller_id="FollowPath"/>'
            "</PipelineSequence>"
            "<ReactiveFallback>"
            "<GoalUpdated/>"
            '<ClearEntireCostmap service_name="global_costmap/clear_entirely_global_costmap"/>'
            "</ReactiveFallback>"
            "</RecoveryNode>"
        )

        self.validator.validate(xml_text)

    def test_accepts_additional_navigation_actions_and_conditions(self) -> None:
        xml_text = tree(
            "<Sequence>"
            '<ComputePathThroughPoses goals="{goals}" path="{path}" planner_id="GridBased"/>'
            '<TruncatePath distance="1.0" input_path="{path}" output_path="{short_path}"/>'
            '<RemovePassedGoals input_goals="{goals}" output_goals="{remaining}" radius="0.5"/>'
            '<Spin spin_dist="1.57"/>'
            '<Wait wait_duration="2.0"/>'
            '<BackUp backup_dist="0.3" backup_speed="0.1"/>'
            '<GoalReached goal="{goal}"/>'
            "<IsStuck/>"
            "</Sequence>"
        )

        self.validator.validate(xml_text)

    def test_rejects_malformed_xml_and_surrounding_text(self) -> None:
        valid_tree = tree('<MoveTo location="A"/>')
        invalid_xml = {
            "malformed": "<root>",
            "leading prose": f"Here is the tree: {valid_tree}",
            "markdown fence": f"```xml\n{valid_tree}\n```",
        }
        for case, xml_text in invalid_xml.items():
            with self.subTest(case=case):
                self.assert_invalid(xml_text, "Malformed XML")

    def test_rejects_invalid_root_and_behavior_tree_structure(self) -> None:
        invalid_xml = {
            "wrong root": '<tree BTCPP_format="4"/>',
            "missing format": tree('<MoveTo location="A"/>', root_attributes=""),
            "wrong format": tree('<MoveTo location="A"/>', root_attributes='BTCPP_format="3"'),
            "unknown root attribute": tree(
                '<MoveTo location="A"/>', root_attributes='BTCPP_format="4" extra="x"'
            ),
            "missing tree id": (
                '<root BTCPP_format="4"><BehaviorTree><MoveTo location="A"/>'
                "</BehaviorTree></root>"
            ),
            "multiple trees": (
                '<root BTCPP_format="4"><BehaviorTree ID="A"><MoveTo location="A"/>'
                '</BehaviorTree><BehaviorTree ID="B"><MoveTo location="B"/>'
                "</BehaviorTree></root>"
            ),
            "multiple tree nodes": tree('<MoveTo location="A"/><MoveTo location="B"/>'),
            "mismatched main tree": tree(
                '<MoveTo location="A"/>',
                root_attributes='BTCPP_format="4" main_tree_to_execute="OtherTree"',
            ),
        }
        for case, xml_text in invalid_xml.items():
            with self.subTest(case=case):
                with self.assertRaises(BTValidationError):
                    self.validator.validate(xml_text)

    def test_rejects_unknown_nodes_and_primitives(self) -> None:
        invalid_xml = {
            "control": tree('<Selector><MoveTo location="A"/></Selector>'),
            "compact action": tree('<Fly destination="A"/>'),
            "explicit action": tree('<Action ID="Fly" destination="A"/>'),
            "explicit condition": tree('<Condition ID="BatteryLow"/>'),
            "missing explicit id": tree("<Action/>"),
        }
        for case, xml_text in invalid_xml.items():
            with self.subTest(case=case):
                with self.assertRaises(BTValidationError):
                    self.validator.validate(xml_text)

    def test_enforces_required_and_allowed_parameters(self) -> None:
        self.assert_invalid(tree("<MoveTo/>"), "Missing required parameter.*location")
        self.assert_invalid(
            tree('<MoveTo location="A" speed="1"/>'), "Unknown parameter.*speed"
        )
        self.assert_invalid(
            tree('<Action ID="MoveTo" location="A" speed="1"/>'),
            "Unknown parameter.*speed",
        )
        self.assert_invalid(
            tree('<ComputePathToPose goal="{goal}" path="{path}"/>'),
            "Missing required parameter.*planner_id",
        )
        self.assert_invalid(
            tree('<FollowPath path="{path}" controller="FollowPath"/>'),
            "Unknown parameter.*controller",
        )

    def test_enforces_node_arity(self) -> None:
        invalid_xml = {
            "empty control": tree("<Sequence/>"),
            "empty decorator": tree("<Inverter/>"),
            "multi-child decorator": tree(
                '<Inverter><MoveTo location="A"/><MoveTo location="B"/></Inverter>'
            ),
            "action child": tree(
                '<MoveTo location="A"><MoveTo location="B"/></MoveTo>'
            ),
        }
        for case, xml_text in invalid_xml.items():
            with self.subTest(case=case):
                with self.assertRaises(BTValidationError):
                    self.validator.validate(xml_text)

    def test_enforces_control_and_decorator_attributes(self) -> None:
        self.validator.validate(
            tree('<Repeat num_cycles="2"><MoveTo location="A"/></Repeat>')
        )
        self.assert_invalid(
            tree(
                '<Parallel><MoveTo location="A"/><MoveTo location="B"/></Parallel>'
            ),
            "Missing required attribute",
        )
        self.assert_invalid(
            tree('<Repeat><MoveTo location="A"/></Repeat>'),
            "Missing required attribute.*num_cycles",
        )
        self.assert_invalid(
            tree('<Sequence unexpected="true"><MoveTo location="A"/></Sequence>'),
            "Unknown attribute.*unexpected",
        )

    def test_rejects_text_inside_tree_elements(self) -> None:
        self.assert_invalid(tree("<Sequence>not XML nodes</Sequence>"), "cannot contain text")

    def test_rejects_missing_malformed_and_inconsistent_vocabularies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            invalid_vocabularies = {
                "malformed.yaml": "control_nodes: [",
                "missing-section.yaml": (
                    "control_nodes: {}\ndecorator_nodes: {}\nactions: {}\n"
                ),
                "wrong-section-type.yaml": (
                    "control_nodes: []\ndecorator_nodes: {}\nactions: {}\nconditions: {}\n"
                ),
                "duplicate-node.yaml": (
                    "control_nodes:\n  Shared: {}\n"
                    "decorator_nodes: {}\n"
                    "actions:\n  Shared: {}\n"
                    "conditions: {}\n"
                ),
                "overlapping-attributes.yaml": (
                    "control_nodes: {}\ndecorator_nodes: {}\n"
                    "actions:\n  MoveTo:\n    required: [location]\n"
                    "    optional: [location]\nconditions: {}\n"
                ),
            }

            for filename, contents in invalid_vocabularies.items():
                with self.subTest(filename=filename):
                    path = directory / filename
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(BTValidationError):
                        BehaviorTreeValidator.from_yaml(path)

            with self.assertRaisesRegex(BTValidationError, "Could not read"):
                BehaviorTreeValidator.from_yaml(directory / "missing.yaml")


if __name__ == "__main__":
    unittest.main()
