from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT
from skeleton_tools.openpose_layouts.face import FACE_LAYOUT
from skeleton_tools.openpose_layouts.hand import HAND_LAYOUT
from skeleton_tools.openpose_layouts.graph_layout import GraphLayout


def combine_layouts(layouts):
    name = 'COMBINED'
    center = layouts['body'].center
    joints = {}
    pairs = []

    last_index = 0
    for n, l in layouts.items():
        joints = {**joints, **{k + last_index: f'{n}_{v}' for (k, v) in l.joints().items()}}
        pairs += [(x1 + last_index, x2 + last_index) for (x1, x2) in l.pairs()]
        last_index += len(l)

    layout = GraphLayout(name, center, joints, pairs, face=True, hand=True, model_pose=layouts['body'].model_pose)
    pairs += [(layout.joint('body_LWrist'), layout.joint('left_hand_Root')), (layout.joint('body_RWrist'), layout.joint('right_hand_Root'))]

    return layout


COMBINED_BODY25_LAYOUT = combine_layouts({'body': BODY_25_LAYOUT, 'face': FACE_LAYOUT, 'left_hand': HAND_LAYOUT, 'right_hand': HAND_LAYOUT})
