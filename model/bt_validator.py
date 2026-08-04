"""Validation for generated BehaviorTree.CPP XML trees."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml


class BTValidationError(ValueError):
    """Raised when a behavior tree or its vocabulary is invalid."""


@dataclass(frozen=True)
class _NodeSpec:
    required: frozenset[str]
    optional: frozenset[str]


class BehaviorTreeValidator:
    """Validate BehaviorTree.CPP XML against a closed YAML vocabulary."""

    _SECTIONS = ("control_nodes", "decorator_nodes", "actions", "conditions")
    _RESERVED_TAGS = {"root", "BehaviorTree", "Action", "Condition"}
    _ROOT_ATTRIBUTES = {"BTCPP_format", "main_tree_to_execute"}

    def __init__(self, vocabulary: dict[str, dict[str, _NodeSpec]]) -> None:
        self._vocabulary = vocabulary

    @classmethod
    def from_yaml(cls, path: str | Path) -> BehaviorTreeValidator:
        """Load and validate a closed node vocabulary from a YAML file."""
        vocabulary_path = Path(path)
        try:
            with vocabulary_path.open("r", encoding="utf-8") as stream:
                raw_vocabulary = yaml.safe_load(stream)
        except OSError as exc:
            raise BTValidationError(
                f"Could not read behavior tree vocabulary '{vocabulary_path}': {exc}"
            ) from exc
        except yaml.YAMLError as exc:
            raise BTValidationError(
                f"Invalid YAML in behavior tree vocabulary '{vocabulary_path}': {exc}"
            ) from exc

        vocabulary = cls._parse_vocabulary(raw_vocabulary)
        return cls(vocabulary)

    @classmethod
    def _parse_vocabulary(cls, raw_vocabulary: Any) -> dict[str, dict[str, _NodeSpec]]:
        if not isinstance(raw_vocabulary, dict):
            raise BTValidationError("Behavior tree vocabulary must be a YAML mapping.")

        unknown_sections = set(raw_vocabulary) - set(cls._SECTIONS)
        missing_sections = set(cls._SECTIONS) - set(raw_vocabulary)
        if unknown_sections:
            names = ", ".join(sorted(map(str, unknown_sections)))
            raise BTValidationError(f"Unknown vocabulary section(s): {names}.")
        if missing_sections:
            names = ", ".join(sorted(missing_sections))
            raise BTValidationError(f"Missing vocabulary section(s): {names}.")

        vocabulary: dict[str, dict[str, _NodeSpec]] = {}
        registered_nodes: set[str] = set()
        for section in cls._SECTIONS:
            raw_nodes = raw_vocabulary[section]
            if not isinstance(raw_nodes, dict):
                raise BTValidationError(f"Vocabulary section '{section}' must be a mapping.")

            nodes: dict[str, _NodeSpec] = {}
            for node_name, raw_spec in raw_nodes.items():
                if not isinstance(node_name, str) or not node_name:
                    raise BTValidationError(f"Node names in '{section}' must be non-empty strings.")
                if node_name in cls._RESERVED_TAGS:
                    raise BTValidationError(f"'{node_name}' is a reserved XML tag.")
                if node_name in registered_nodes:
                    raise BTValidationError(f"Node '{node_name}' is registered more than once.")

                nodes[node_name] = cls._parse_node_spec(node_name, raw_spec)
                registered_nodes.add(node_name)
            vocabulary[section] = nodes

        return vocabulary

    @staticmethod
    def _parse_node_spec(node_name: str, raw_spec: Any) -> _NodeSpec:
        if raw_spec is None:
            raw_spec = {}
        if not isinstance(raw_spec, dict):
            raise BTValidationError(f"Vocabulary entry for '{node_name}' must be a mapping.")

        unknown_keys = set(raw_spec) - {"required", "optional"}
        if unknown_keys:
            names = ", ".join(sorted(map(str, unknown_keys)))
            raise BTValidationError(f"Unknown setting(s) for '{node_name}': {names}.")

        required = BehaviorTreeValidator._parse_attribute_list(node_name, "required", raw_spec)
        optional = BehaviorTreeValidator._parse_attribute_list(node_name, "optional", raw_spec)
        overlap = required & optional
        if overlap:
            names = ", ".join(sorted(overlap))
            raise BTValidationError(
                f"Attribute(s) cannot be both required and optional for '{node_name}': {names}."
            )
        if "name" in required:
            raise BTValidationError(
                f"The standard 'name' attribute cannot be required for '{node_name}'."
            )

        return _NodeSpec(frozenset(required), frozenset(optional))

    @staticmethod
    def _parse_attribute_list(
        node_name: str, key: str, raw_spec: dict[str, Any]
    ) -> set[str]:
        values = raw_spec.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise BTValidationError(
                f"'{key}' for '{node_name}' must be a list of non-empty strings."
            )
        if len(values) != len(set(values)):
            raise BTValidationError(f"'{key}' for '{node_name}' contains duplicates.")
        return set(values)

    def validate(self, xml_text: str) -> ET.Element:
        """Parse and validate one complete generated XML response."""
        if not isinstance(xml_text, str) or not xml_text.strip():
            raise BTValidationError("Behavior tree XML must be a non-empty string.")

        try:
            root = ET.fromstring(xml_text.strip())
        except ET.ParseError as exc:
            raise BTValidationError(f"Malformed XML: {exc}.") from exc

        if root.tag != "root":
            raise BTValidationError(f"Expected root tag <root>, found <{root.tag}>.")
        self._validate_text(root)
        self._validate_root_attributes(root)

        behavior_trees = list(root)
        if len(behavior_trees) != 1 or behavior_trees[0].tag != "BehaviorTree":
            raise BTValidationError("<root> must contain exactly one <BehaviorTree> element.")

        behavior_tree = behavior_trees[0]
        self._validate_text(behavior_tree)
        if set(behavior_tree.attrib) != {"ID"} or not behavior_tree.attrib["ID"]:
            raise BTValidationError(
                "<BehaviorTree> must have exactly one non-empty 'ID' attribute."
            )

        main_tree = root.attrib.get("main_tree_to_execute")
        if main_tree is not None and main_tree != behavior_tree.attrib["ID"]:
            raise BTValidationError(
                "'main_tree_to_execute' must match the <BehaviorTree> ID."
            )

        tree_nodes = list(behavior_tree)
        if len(tree_nodes) != 1:
            raise BTValidationError("<BehaviorTree> must contain exactly one tree node.")

        self._validate_node(tree_nodes[0])
        return root

    def _validate_root_attributes(self, root: ET.Element) -> None:
        unknown = set(root.attrib) - self._ROOT_ATTRIBUTES
        if unknown:
            names = ", ".join(sorted(unknown))
            raise BTValidationError(f"Unknown <root> attribute(s): {names}.")
        if root.attrib.get("BTCPP_format") != "4":
            raise BTValidationError("<root> must declare BTCPP_format=\"4\".")

    def _validate_node(self, node: ET.Element) -> None:
        self._validate_text(node)
        node_name, section, reserved_attributes = self._resolve_node(node)
        spec = self._vocabulary[section][node_name]
        self._validate_attributes(node, node_name, section, spec, reserved_attributes)

        child_count = len(node)
        if section == "control_nodes" and child_count < 1:
            raise BTValidationError(f"Control node '{node_name}' must have at least one child.")
        if section == "decorator_nodes" and child_count != 1:
            raise BTValidationError(f"Decorator node '{node_name}' must have exactly one child.")
        if section in {"actions", "conditions"} and child_count != 0:
            raise BTValidationError(f"Leaf node '{node_name}' cannot have children.")

        for child in node:
            self._validate_node(child)

    def _resolve_node(self, node: ET.Element) -> tuple[str, str, set[str]]:
        if node.tag in {"Action", "Condition"}:
            section = "actions" if node.tag == "Action" else "conditions"
            node_name = node.attrib.get("ID", "")
            if not node_name:
                raise BTValidationError(f"Explicit <{node.tag}> node requires a non-empty 'ID'.")
            if node_name not in self._vocabulary[section]:
                raise BTValidationError(f"Unknown {node.tag.lower()} primitive '{node_name}'.")
            return node_name, section, {"ID"}

        for section in self._SECTIONS:
            if node.tag in self._vocabulary[section]:
                return node.tag, section, set()

        raise BTValidationError(f"Unknown behavior tree node '{node.tag}'.")

    @staticmethod
    def _validate_attributes(
        node: ET.Element,
        node_name: str,
        section: str,
        spec: _NodeSpec,
        reserved_attributes: set[str],
    ) -> None:
        present = set(node.attrib) - reserved_attributes - {"name"}
        unknown = present - spec.required - spec.optional
        if unknown:
            label = "parameter" if section in {"actions", "conditions"} else "attribute"
            names = ", ".join(sorted(unknown))
            raise BTValidationError(f"Unknown {label}(s) for '{node_name}': {names}.")

        missing = spec.required - present
        if missing:
            label = "parameter" if section in {"actions", "conditions"} else "attribute"
            names = ", ".join(sorted(missing))
            raise BTValidationError(f"Missing required {label}(s) for '{node_name}': {names}.")

    @staticmethod
    def _validate_text(element: ET.Element) -> None:
        if element.text and element.text.strip():
            raise BTValidationError(f"Element <{element.tag}> cannot contain text.")
        for child in element:
            if child.tail and child.tail.strip():
                raise BTValidationError(f"Element <{element.tag}> cannot contain text.")
