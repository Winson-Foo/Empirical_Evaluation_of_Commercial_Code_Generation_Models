"""Functions which process model output bytes to make them ready for eval.

Note: postprocessors must either accept an `example` and `is_target` kwargs
or include `**unused_kwargs` in their signature. The `example` will be the
full example.

These functions should assume input strings to be unicode, but that strings
in the `example` dict will be in bytes.
"""

from typing import Any, Dict, List, Tuple, Union


def string_to_float(string: str, default: float = -1., **unused_kwargs: Any) -> float:
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def lower_text(string: str, **unused_kwargs: Any) -> str:
    """Lowercases text."""
    return string.lower()


def string_label_to_class_id(
    string_label: str,
    label_classes: List[str],
    default: int = -1,
    **unused_kwargs: Any
) -> int:
    """Returns index of string_label in label_classes or default if not found."""
    if string_label in label_classes:
        return label_classes.index(string_label)
    else:
        return default


def multirc(
    string_label: str,
    example: Dict[str, Union[bytes, str]] = None,
    is_target: bool = False
) -> Dict[str, Union[int, str]]:
    """Returns dict containing the class with the question index for grouping."""
    res = {
        "value": string_label_to_class_id(
            string_label, example=example, label_classes=("False", "True")
        )
    }
    # Add the group, if present, since the model outputs will not have it.
    if is_target:
        res["group"] = example["idx/question"]
    return res


def record(
    answer: Any,
    example: Dict[str, Union[bytes, List[bytes]]] = None,
    is_target: bool = False
) -> Union[List[str], Dict[str, Union[List[str], Tuple[int, int]]]]:
    """Returns dict with answer, or all answers + grouping key for a target."""
    if is_target:
        return {
            "value": [a.decode("utf-8") for a in example["answers"]],
            # Add the group since the model output will not have it.
            "group": (example["idx/passage"], example["idx/query"])
        }
    return {"value": answer}


def qa(
    answer: Any,
    example: Dict[str, Union[bytes, List[bytes]]] = None,
    is_target: bool = False
) -> Union[Any, List[str]]:
    """Returns answer, or all answers if the full example is provided."""
    if is_target:
        return [a.decode("utf-8") for a in example["answers"]]
    return answer


def span_qa(
    answer: Any,
    example: Dict[str, Union[bytes, str]] = None,
    is_target: bool = False
) -> Union[Any, Dict[str, List[str]]]:
    """Returns answer, or a dict with answers and context if the example is provided."""
    if is_target:
        return {
            "answers": [a.decode("utf-8") for a in example["answers"]],
            "context": example["context"].decode("utf-8")
        }
    return answer


def wsc_simple(
    prediction: str,
    example: Dict[str, Union[bytes, str]] = None,
    is_target: bool = False
) -> int:
    """Sees whether we predicted the referent or not."""
    if is_target:
        return example["label"]

    determiners = {
        "a", "an", "few", "her", "his", "each", "every", "many", "much", "my",
        "our", "some", "that", "the", "their", "these", "this", "those", "which",
        "whose", "your"
    }

    def clean(s: str) -> str:
        """Ignore capitalization and determiners."""
        s = s.strip().lower()
        return " ".join([w for w in s.split(" ") if w not in determiners])

    prediction = clean(prediction)
    if not prediction:
        # We don't want an empty prediction to accidentally return 0 and spuriously
        # match the label.
        return -1

    # We aren't using the label but rather using the extracted referent so that we
    # can see if the prediction is equivalent to the referent.
    referent = clean(example["targets_pretokenized"].decode("utf-8"))

    if ("'" in prediction) != ("'" in referent):
        # Make sure we don't mark cases where the prediction is "Bob" and the
        # referent is "Bob's hat" as predicting the referent.
        predicted_referent = False
    else:
        prediction_words = set(prediction.split(" "))
        referent_words = set(referent.split(" "))

        # Handle cases where the prediction is "fuzzy bunny" and the referent is
        # "bunny".
        predicted_referent = prediction_words.issubset(
            referent_words) or referent_words.issubset(prediction_words)

    return int(predicted_referent)


def rank_classification(
    score: Any,
    example: Dict[str, Union[bytes, int, float, List[bytes]]] = None,
    is_target: bool = False,
    passthrough_feature_keys: List[str] = None
) -> Union[Any, Tuple[Tuple[int, int], bool, float, int]]:
    """A postprocessor for the `rank_classification` preprocessor and metric."""
    if is_target:
        outputs = [
            tuple(example["idx"]), example["is_correct"],
            example.get("weight", 1.0),
            len(example["targets"])
        ]
        if passthrough_feature_keys:
            for key in passthrough_feature_keys:
                outputs.append(example[key])
        return tuple(outputs)
    else:
        return score