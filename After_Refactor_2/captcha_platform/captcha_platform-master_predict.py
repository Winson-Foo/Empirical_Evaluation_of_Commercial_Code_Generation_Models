#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
from typing import List, Dict
from config import ModelConfig


class ModelUtil:
    @staticmethod
    def decode_maps(categories: List[str]) -> Dict[int, str]:
        return {index: category for index, category in enumerate(categories, 0)}


def predict_func(
        image_batch: List, 
        _sess, 
        dense_decoded, 
        op_input, 
        model: ModelConfig, 
        output_split: str = None,
) -> str:

    output_split = model.output_split if output_split is None else output_split
    category_split = model.category_split if model.category_split else ""

    dense_decoded_code = _sess.run(dense_decoded, feed_dict={
        op_input: image_batch,
    })

    decoded_expression = [
        category_split.join([
            "" if i == -1 or i == model.category_num else ModelUtil.decode_maps(model.category)[i]
            for i in item
        ])
        for item in dense_decoded_code
    ]

    return output_split.join(decoded_expression) if len(decoded_expression) > 1 else decoded_expression[0]