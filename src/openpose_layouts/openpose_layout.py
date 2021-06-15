class OpenPoseLayout:
    def __init__(self, name, center, joints, pairs):
        self.name = name
        self.center = center
        self._pose_joints = joints
        self._pose_pairs = pairs
        self._len = len(joints)
        self._pose_map = {}
        for k, v in joints.items():
            self._pose_map[k] = v
            self._pose_map[v] = k

    def __len__(self):
        return self._len

    def joint(self, i):
        return self._pose_map[i]

    def joints(self):
        return self._pose_joints

    def pairs(self):
        return self._pose_pairs
