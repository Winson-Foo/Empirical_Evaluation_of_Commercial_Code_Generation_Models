#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
from typing import List, Dict
from config import ModelConfig


def decode_maps(categories: List[str]) -> Dict[int, str]:
    return {index: category for index, category in enumerate(categories, 0)}


def decode_expression(item: List[int], model: ModelConfig) -> str:
    expression = []
    for i in item:
        if i == -1 or i == model.category_num:
            expression.append("")
        else:
            expression.append(decode_maps(model.category)[i])
    return model.category_split.join(expression)


def predict_func(image_batch: Any, _sess: Any, dense_decoded: Any, op_input: Any, model: ModelConfig, 
                 output_split: str = None) -> str:

    output_split = model.output_split if output_split is None else output_split

    dense_decoded_code = _sess.run(dense_decoded, feed_dict={
        op_input: image_batch,
    })
    
    decoded_expression = [decode_expression(item, model) for item in dense_decoded_code]
    
    return output_split.join(decoded_expression) if len(decoded_expression) > 1 else decoded_expression[0]