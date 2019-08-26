import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

def py_func(func,
            args=(),
            kwargs={},
            output_types=None,
            output_shapes=None,
            name=None):
    if not isinstance(args, (list, tuple)):
        raise TypeError('args must be list and not {}. args: {}'.format(type(args), args))

    if not isinstance(kwargs, dict):
        raise TypeError('kwargs must be dict and not {}. args: {}'.format(type(kwargs), kwargs))


    # For dynamic type inference use callable output_types and output_shapes
    if callable(output_types):
        # If callable, assume same signature and call with tensors and get the types
        output_types = output_types(*args, **kwargs)
    if callable(output_shapes):
        # If callable, assume same signature and call with tensors and get the shapes
        output_shapes = output_shapes(*args, **kwargs)

    flat_output_types = nest.flatten(output_types)
    args = (args, kwargs)
    flat_args = nest.flatten(args)


    def python_function_wrapper(*py_args):
        py_args, py_kwargs = nest.pack_sequence_as(args, py_args)

        ret = func(*py_args, **py_kwargs)
        # ToDo: Catch Exceptions and improve msg, because tensorflow ist not able
        # to preserve the traceback, i.e. the Exceptions does not contain any
        # information where the Exception was raised.
        nest.assert_shallow_structure(output_types, ret)
        return nest.flatten(ret)

    flat_values = tf.py_function(python_function_wrapper, flat_args, flat_output_types, name=name)

    if output_shapes is not None:
        # I am not sure if this is nessesary
        output_shapes = nest.map_structure_up_to(output_types, tensor_shape.as_shape, output_shapes)

    flattened_shapes = nest.flatten(output_shapes)
    for ret_t, shape in zip(flat_values, flattened_shapes):
        ret_t.set_shape(shape)

    return nest.pack_sequence_as(output_types, flat_values)

def from_indexable(iterator, output_types, output_shapes, num_parallel_calls=None, name=None):
    ds = tf.data.Dataset.range(len(iterator))

    def index_to_entry(index):
        return py_func(
                func=iterator.__getitem__,
                args=(index,),
                output_types=output_types,
                output_shapes=output_shapes,
                name=name)

    return ds.map(index_to_entry, num_parallel_calls=num_parallel_calls)


