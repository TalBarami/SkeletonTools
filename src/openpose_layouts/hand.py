from openpose_layouts.openpose_layout import OpenPoseLayout

HAND_LAYOUT = OpenPoseLayout(
    'HAND',
    9,
    {0: 'Root',
     1: 'Thumb1CMC',
     2: 'Thumb2Knuckles',
     3: 'Thumb3IP',
     4: 'Thumb4FingerTip',
     5: 'Index1Knuckles',
     6: 'Index2PIP',
     7: 'Index3DIP',
     8: 'Index4FingerTip',
     9: 'Middle1Knuckles',
     10: 'Middle2PIP',
     11: 'Middle3DIP',
     12: 'Middle4FingerTip',
     13: 'Ring1Knuckles',
     14: 'Ring2PIP',
     15: 'Ring3DIP',
     16: 'Ring4FingerTip',
     17: 'Pinky1Knuckles',
     18: 'Pinky2PIP',
     19: 'Pinky3DIP',
     20: 'Pinky4FingerTip'},
    [(0, 1), (1, 2), (2, 3), (3, 4),
     (0, 5), (5, 6), (6, 7), (7, 8),
     (0, 9), (9, 10), (10, 11), (11, 12),
     (0, 13), (13, 14), (14, 15), (15, 16),
     (0, 17), (17, 18), (18, 19), (19, 20)]
)
