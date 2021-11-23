__all__ = ["poly_encoder", "transformer"]


# def convert_tf_datasets(
#     dataset,
#     input_key: Union[str, List[str], Tuple[str]],
#     labels_key: Union[str, List[str], Tuple[str]],
#     dtype: Literal["dict", "tuple"] = "tuple",
# ) -> tf.data.Dataset:
#     if dtype not in ["dict", "tuple"]:
#         raise ValueError("dtype argument must be a 'dict' or 'tuple'.")

#     x = dataset[input_key]
#     y = dataset[labels_key]

#     if isinstance(input_key, str):
#         x_signature = get_signature(x)
#     elif dtype == "dict":
#         x_signature = {key: get_signature(dataset[key]) for key in input_key}
#     elif dtype == "tuple":
#         x_signature = tuple([get_signature(dataset[key]) for key in input_key])

#     if isinstance(labels_key, str):
#         y_signature = get_signature(y)
#     elif dtype == "dict":
#         y_signature = {key: get_signature(dataset[key]) for key in labels_key}
#     elif dtype == "tuple":
#         y_signature = tuple([get_signature(dataset[key]) for key in labels_key])

#     def _generate():
#         for batch_x, batch_y in zip(x, y):
#             if dtype == "tuple":
#                 if not isinstance(input_key, str):
#                     batch_x = tuple(batch_x.values())
#                 if not isinstance(labels_key, str):
#                     batch_y = tuple(batch_y.values())

#             yield batch_x, batch_y

#     tf_dataset = tf.data.Dataset.from_generator(
#         _generate, output_signature=(x_signature, y_signature)
#     )
#     tf_dataset = tf_dataset.apply(tf.data.experimental.assert_cardinality(len(x)))

#     return tf_dataset
