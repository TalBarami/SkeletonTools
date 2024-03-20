class GraphLayout:
    def __init__(self, name, center, joints, pairs, face=False, hand=False, model_pose=None):
        self.name = name
        self.center = center
        self._pose_joints = joints
        self._pose_pairs = pairs
        self._len = len(joints)
        self._pose_map = {}
        self.model_pose = model_pose if model_pose else name
        self.face = face
        self.hand = hand
        for k, v in joints.items():
            self._pose_map[k] = v
            self._pose_map[v] = k

    def face_joints(self):
        return [k for k in self._pose_map.keys() if type(k) == str and any(s in k for s in ['Eye', 'Ear', 'Nose'])]

    def __len__(self):
        return self._len

    def __call__(self, *args, **kwargs):
        return self

    def joint(self, i):
        return self._pose_map[i]

    def joints(self):
        return self._pose_joints

    def pairs(self):
        return self._pose_pairs

    def neighbors(self, i):
        return []


def convert_layout(np_data, l1, l2):
    assert len(l2) <= len(l1)
    joints = [l1.joint(i) for i in l2.joints().values()]
    return np_data[joints]
