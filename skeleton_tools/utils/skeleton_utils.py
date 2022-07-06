from copy import deepcopy

import numpy as np
import random
import pandas as pd
from scipy.interpolate import interp1d
import scipy.fftpack as fp

from tqdm import tqdm

from skeleton_tools.utils.constants import EPSILON, JSON_SOURCES


def get_last_frame_id(data_numpy):
    _, T, _, _ = data_numpy.shape
    last_frame = 0
    for i in range(T):
        last_frame = i
        if not np.any(data_numpy[:, i:, :, :]):
            break
    return last_frame


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M), dtype=data_numpy.dtype)
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def random_positioning(data_numpy):
    x = np.copy(data_numpy)
    C, T, V, M = data_numpy.shape

    for m in range(M):
        intervals = {
            0: (-0.5 - np.clip(x[0, :, :, m].min(), -0.5, 1), 0.5 - np.clip(x[0, :, :, m].max(), -1, 0.5)),
            1: (-0.5 - np.clip(x[1, :, :, m].min(), -0.5, 1), 0.5 - np.clip(x[1, :, :, m].max(), -1, 0.5)),
        }

        for a in range(2):
            x[a, :, :, m] += np.random.uniform(*intervals[a])
    return x


def mirror_sample(data_numpy):
    x = np.copy(data_numpy)
    x[0] = -x[0]
    return x


def reverse_sample(data_numpy):
    last_frame = get_last_frame_id(data_numpy)
    return np.concatenate([np.flip(data_numpy[:, :last_frame, :, :], axis=1), data_numpy[:, last_frame:, :, :]], axis=1)


def pad_reversed(data_numpy):
    _, T, _, _ = data_numpy.shape
    T -= 1
    padded = np.copy(data_numpy)
    while True:
        last_frame = get_last_frame_id(padded)
        padded = np.concatenate(padded[:, :last_frame, :, :], np.flip(padded[:, :np.min((2 * last_frame, T)), :, :]), axis=1)
        if last_frame == T:
            break


def random_repetition(data_numpy, rep_generator):
    label, amplitude, cycle = rep_generator()
    if label:
        data_numpy = add_repetitive_noise(data_numpy, amplitude, cycle, 0)
    return data_numpy, label


def apply_noise(X, c, amplitude, cycle, offset, size):
    r = X.copy()
    F = np.arange(X.shape[0])
    r += amplitude * size * np.sin(offset + 2 * np.pi * F / cycle)
    r[c < EPSILON] = 0
    return np.clip(np.round(r, 8), -1, 1)


def get_interval(c, w, h):
    a = np.array(c) + np.array([w, h])
    b = np.array(c) - np.array([w, h])
    return np.linalg.norm(a-b)


def add_repetitive_noise(data_numpy, amplitude, cycle, offset):
    _, F, _, M = data_numpy.shape
    joints = [0, 4, 7]
    axis = 1

    for p in range(M):
        lengths = np.array([get_interval(bounding_box(data_numpy[:2, f, :, p], data_numpy[2, f, :, p])) for f in range(F)])

        for j in joints:
            v = data_numpy[axis, :, j, p]
            c = data_numpy[2, :, j, p]
            data_numpy[axis, :, j, p] = apply_noise(v, c, amplitude, cycle, offset, lengths)

    return data_numpy


def centralize_json(pose_json):
    result = deepcopy(pose_json)
    boxes = [bounding_box((s['pose'][::2], s['pose'][1::2]), s['pose_score']) for s in [d['skeleton'][0] for d in result if len(d['skeleton']) > 0]]
    center, size = None, None
    if any(boxes):
        try:
            size = np.median(np.array([s for c, s in boxes]).T, axis=1)
            center, _ = boxes[0]
        except Exception as e:
            print('Error')
    for d in tqdm(result, ascii=True, desc='Centralizing'):
        for s in d['skeleton']:
            if center is None:
                center, _ = bounding_box((s['pose'][::2], s['pose'][1::2]), s['pose_score'])
            if size is None:
                _, size = bounding_box((s['pose'][::2], s['pose'][1::2]), s['pose_score'])
            for src in [src for src in JSON_SOURCES if src['name'] in s.keys()]:
                key = src['name']
                c = np.array(s[f'{key}_score'])
                xy = np.array([s[key][::2], s[key][1::2]]).T
                xy -= center
                xy /= size
                xy[c < EPSILON] = 0
                s[src['name']] = [e for l in zip(xy[:, 0], xy[:, 1]) for e in l]
    return result

def decentralize_json(pose_json, resolution):
    result = deepcopy(pose_json)
    for d in tqdm(result, ascii=True, desc='Centralizing'):
        for s in d['skeleton']:
            for src in [src for src in JSON_SOURCES if src['name'] in s.keys()]:
                key = src['name']
                c = np.array(s[f'{key}_score'])
                xy = np.array([s[key][::2], s[key][1::2]]).T
                xy += resolution / 2
                xy *= resolution / 10
                xy[c < EPSILON] = 0
                s[src['name']] = [e for l in zip(xy[:, 0], xy[:, 1]) for e in l]
    return result


def normalize_json(pose_json, resolution, centralize=True):
    result = deepcopy(pose_json)
    width, height = resolution
    for d in tqdm(result, ascii=True, desc='Normalizing & Centralizing'):
        for s in d['skeleton']:
            for src in [src for src in JSON_SOURCES if src['name'] in s.keys()]:
                key = src['name']
                c = np.array(s[f'{key}_score'])
                xy = np.array([s[key][::2], s[key][1::2]])
                xy = np.round((xy.T / np.array([width, height])).T - 0.5 if centralize else 0, 8)
                xy.T[c < EPSILON] = 0
                x, y = xy[0], xy[1]
                s[src['name']] = [e for l in zip(x, y) for e in l]
    return result

def denormalize_json(pose_json, resolution, centralized=True):
    result = deepcopy(pose_json)
    for d in tqdm(result, ascii=True, desc='Normalizing & Centralizing'):
        for s in d['skeleton']:
            for src in [src for src in JSON_SOURCES if src['name'] in s.keys()]:
                key = src['name']
                c = np.array(s[f'{key}_score'])
                xy = (np.array([s[key][::2], s[key][1::2]]) + (0.5 if centralized else 0)).T
                xy *= resolution
                xy[c < EPSILON] = 0
                s[src['name']] = [e for l in zip(xy[:, 0], xy[:, 1]) for e in l]
    return result

def denormalize_numpy(data_numpy, resolution):
    x = data_numpy.copy()
    width, height = resolution
    x[:2] += 0.5
    x[0][x[2] < EPSILON] = 0
    x[1][x[2] < EPSILON] = 0
    x[0] *= width
    x[1] *= height
    return x

def to_fft(data_numpy):
    ft = fp.fft(data_numpy[0, :, :, :] + 1j * data_numpy[1, :, :, :], axis=2)
    return np.concatenate((np.expand_dims(ft.real, axis=0), np.expand_dims(ft.imag, axis=0)))

# def add_repetitive_noise_json(json_file, amplitude, cycle, offset):
#     j_copy = deepcopy(json_file)
#     j_copy['data'] = j_copy['data'][1:]
#     pose = [f['skeleton'][0]['pose'] for f in j_copy['data']]
#     xyc = np.array([
#         np.array([p[::2] for p in pose]),
#         np.array([p[1::2] for p in pose]),
#         np.array([f['skeleton'][0]['score'] for f in j_copy['data']])
#     ])
#     _, F, _ = xyc.shape
#     axis = 1
#     joints = [4, 7]
#     for j in joints:
#         intervals = np.array([get_interval(bounding_box(xyc[:2, f, :], xyc[2, f, :]), axis) for f in range(F)])
#         xyc[axis, :, j] = apply_noise(xyc[axis, :, j], xyc[2, :, j], amplitude, cycle, offset, intervals)
#     for i, p in enumerate(pose):
#         p[0::2] = xyc[0, i, :]
#         p[1::2] = xyc[1, i, :]
#     return j_copy


# def bounding_box2(pose, score):
#     x, y = pose[0][score > EPSILON], pose[1][score > EPSILON]
#     if not any(x):
#         x = np.array([0])
#     if not any(y):
#         y = np.array([0])
#     box = {
#         0: {
#             'min': np.min(x),
#             'max': np.max(x),
#         },
#         1: {
#             'min': np.min(y),
#             'max': np.max(y)
#         }
#     }
#     return box

def box_distance(b1, b2):
    c1, _ = b1
    c2, _ = b2
    return np.linalg.norm(c1 - c2)

def bounding_box(pose, score):
    pose, score = np.array(pose), np.array(score)
    x, y = pose[0][score > EPSILON], pose[1][score > EPSILON]
    if not any(x):
        x = np.array([0])
    if not any(y):
        y = np.array([0])
    w, h = (np.max(x) - np.min(x)), (np.max(y) - np.min(y))
    return np.array((np.min(x) + w / 2, np.min(y) + h / 2)), np.array((w, h))



def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :,
                                     forward_map[t]].transpose(
            1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def interpolate(data_numpy):
    C, T, V, M = data_numpy.shape
    x = np.copy(data_numpy)
    for v in range(V):
        for m in range(M):
            scores = data_numpy[2, :, v, m]
            if (scores < 0.4).mean() > 0.3:
                continue
            for c in range(C - 1):
                # y = data_numpy[c, :, v, m]
                # y[scores < 0.4] = np.nan
                # x[c, :, v, m] = interp1d(np.arange(len(y)), y, kind='cubic')
                y = data_numpy[c, :, v, m]
                s = pd.Series(y)
                s[scores < 0.2] = np.nan
                y = s.interpolate(method='spline', order=2, limit_direction='both').to_numpy()
                x[c, :, v, m] = y
    return x