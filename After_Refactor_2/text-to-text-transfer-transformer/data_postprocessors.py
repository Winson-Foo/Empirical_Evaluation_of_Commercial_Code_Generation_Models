import tensorflow.compat.v2 as tf


def string_to_float(string, default=-1.):
  try:
    return float(string)
  except ValueError:
    return default


def lower_case(string):
  return tf.compat.as_text(string).lower()


def string_label_to_class_id(string_label, label_classes, default=-1):
  if string_label in label_classes:
    return label_classes.index(string_label)
  else:
    return default


def multirc(string_label, example=None, is_target=False):
  res = {
      "value": string_label_to_class_id(
          string_label, example=example, label_classes=("False", "True"))
  }
  if is_target:
    res["group"] = example["idx/question"]
  return res


def record(answer, example=None, is_target=False):
  if is_target:
    return {
        "value": [tf.compat.as_text(a) for a in example["answers"]],
        "group": (example["idx/passage"], example["idx/query"])
    }
  return {"value": answer}


def qa(answer, example=None, is_target=False):
  if is_target:
    return [tf.compat.as_text(a) for a in example["answers"]]
  return answer


def span_qa(answer, example=None, is_target=False):
  if is_target:
    return {
        "answers": [tf.compat.as_text(a) for a in example["answers"]],
        "context": tf.compat.as_text(example["context"])
    }
  return answer


def wsc_simple(prediction, example=None, is_target=False):
  if is_target:
    return example["label"]

  determiners = {
      "a", "an", "few", "her", "his", "each", "every", "many", "much", "my",
      "our", "some", "that", "the", "their", "these", "this", "those", "which",
      "whose", "your"
  }

  def clean(string):
    string = tf.compat.as_text(string).strip().lower()
    return " ".join([word for word in string.split(" ") if word not in determiners])

  prediction = clean(prediction)
  if not prediction:
    return -1

  referent = clean(example["targets_pretokenized"])

  if ("'" in prediction) != ("'" in referent):
    predicted_referent = False
  else:
    prediction_words = set(prediction.split(" "))
    referent_words = set(referent.split(" "))

    predicted_referent = prediction_words.issubset(referent_words) or referent_words.issubset(prediction_words)

  return int(predicted_referent)


def rank_classification(score, example=None, is_target=False, passthrough_feature_keys=None):
  if is_target:
    outputs = [
        tuple(example["idx"]), example["is_correct"], example.get("weight", 1.0),
        len(example["targets"])
    ]
    if passthrough_feature_keys:
      for key in passthrough_feature_keys:
        outputs.append(example[key])
    return tuple(outputs)
  else:
    return score