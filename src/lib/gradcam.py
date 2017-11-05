import cv2
import inspect
import numpy as np
import weakref

import chainer
from chainer import function_hook
from chainer import cuda


class CallerInspectionHook(function_hook.FunctionHook):

    name = 'CallerInspectionHook'

    def __init__(self):
        self._call_history = []

    def forward_postprocess(self, function, in_data):
        stack = inspect.stack()
        function_node = stack[1][0].f_locals['self']
        # function that calls "apply" of FunctionNode instance
        function = stack[2][0].f_globals[stack[2][3]]
        # serch for Link instance that calls Chainer function
        link = None
        for s in stack[3:]:
            frame = s[0]
            if frame.f_locals.has_key('self'):
                caller_instance = frame.f_locals['self']
                if isinstance(caller_instance, chainer.Link):
                    link = caller_instance
                    break
        self._call_history.append({
            'function_node': function_node,
            'function': function,
            'link': link,
        })

    @property
    def call_history(self):
        return self._call_history


def _call_is_matched(function_node, call, node_to_item):
    if isinstance(call, chainer.Link):
        return call == node_to_item[function_node]['link']
    return call == node_to_item[function_node]['function']


def _sequence_is_matched(function_node, call_sequence, node_to_item):
    head = call_sequence[0]
    if not _call_is_matched(function_node, head, node_to_item):
        return False
    tail = call_sequence[1:]
    if tail == []:
        return True
    for x in function_node.inputs:
        if x.creator_node is None:
            continue
        if _sequence_is_matched(x.creator_node, tail, node_to_item):
            return True
    return False


def _find_call_sequence(call_history, call_sequence):
    reversed_sequence = list(reversed(call_sequence))
    node_to_item = {item['function_node']: item for item in call_history}
    for i, history_item in enumerate(call_history):
        function_node = history_item['function_node']
        if _sequence_is_matched(function_node, reversed_sequence, node_to_item):
            return i, function_node
    return -1, None


def gradcam(model, x, calls, y_grad=None, loss_func=None, feature_index=0):
    if isinstance(x, chainer.Variable):
        xp = cuda.get_array_module(x.data)
    else:
        xp = cuda.get_array_module(x)

    with CallerInspectionHook() as hook:
        with chainer.using_config('train', False):
            if loss_func is not None:
                y = loss_func(x)
            else:
                y = model(x)
        index, function_node = _find_call_sequence(hook.call_history, calls)
        # store variables because their gradients vanishes without reference to them
        h_node = function_node.outputs[feature_index]()
        h = h_node.get_variable()
        h_node._variable = weakref.ref(h)

    model.cleargrads()
    y.grad = y_grad
    y.backward(True, True)

    weights = xp.mean(h.grad, axis=tuple(np.arange(2, h.ndim)), keepdims=True)
    weights = xp.broadcast_to(weights, h.shape)
    gcam = xp.sum(weights * h.data, axis=1)
    max_gcam = xp.max(gcam, axis=tuple(np.arange(1, gcam.ndim)), keepdims=True)
    max_gcam = xp.broadcast_to(max_gcam, gcam.shape)
    gcam = (gcam > 0) * gcam / max_gcam
    gcam = (gcam * 256).clip(0, 255).astype(np.uint8)
    return gcam

def heatmap(gcam, output_size=None, colormap=cv2.COLORMAP_JET):
    gcam = np.uint8(gcam)
    if output_size is not None:
        gcam = cv2.resize(gcam, output_size)
    h = cv2.applyColorMap(gcam, colormap)
    return h

def overlay(image, gcam, colormap=cv2.COLORMAP_JET):
    height, width, _ = image.shape
    gcam = cv2.resize(gcam, (width, height))
    h = heatmap(gcam, colormap=colormap)
    h = np.float32(h)
    image = np.float32(image)
    w = gcam[:,:,np.newaxis] / np.float32(gcam.max())
    out_image = (image * (1 - w) + h * w)
    return np.uint8(out_image)
