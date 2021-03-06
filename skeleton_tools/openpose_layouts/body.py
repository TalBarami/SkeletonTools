from skeleton_tools.openpose_layouts.graph_layout import GraphLayout

BODY_25_LAYOUT = GraphLayout(
    'BODY_25',
    1,
    {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "MidHip",
        9: "RHip",
        10: "RKnee",
        11: "RAnkle",
        12: "LHip",
        13: "LKnee",
        14: "LAnkle",
        15: "REye",
        16: "LEye",
        17: "REar",
        18: "LEar",
        19: "LBigToe",
        20: "LSmallToe",
        21: "LHeel",
        22: "RBigToe",
        23: "RSmallToe",
        24: "RHeel"
        # 25: "Background"
    }, [(0, 1), (1, 8),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11), (11, 22), (11, 24), (22, 23),
        (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20),
        (0, 15), (15, 17), (0, 16), (16, 18)]
)

BODY_21_LAYOUT = GraphLayout(
    'BODY_21',
    1,
    {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "LowerAbs",
        9: "RHip",
        10: "RKnee",
        11: "RAnkle",
        12: "LHip",
        13: "LKnee",
        14: "LAnkle",
        15: "REye",
        16: "LEye",
        17: "REar",
        18: "LEar",
        19: "RealNeck",
        20: "Top"
        # 21: "Background"
    }, [(1, 8), (0, 1), (1, 19), (19, 20),
        (0, 15), (15, 17),
        (0, 16), (16, 18),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (2, 9), (8, 9), (9, 10), (10, 11),
        (5, 12), (8, 12), (12, 13), (13, 14),
        (2, 17), (5, 18)
        ]
)

COCO_LAYOUT = GraphLayout(
    'COCO',
    1,
    {
        0: "Nose",
        1: "LEye",
        2: "REye",
        3: "LEar",
        4: "REar",
        5: "LShoulder",
        6: "RShoulder",
        7: "LElbow",
        8: "RElbow",
        9: "LWrist",
        10: "RWrist",
        11: "LHip",
        12: "RHip",
        13: "LKnee",
        14: "RKnee",
        15: "LAnkle",
        16: "RAnkle"
        # 0: "Nose",
        # 1: "Neck",
        # 2: "RShoulder",
        # 3: "RElbow",
        # 4: "RWrist",
        # 5: "LShoulder",
        # 6: "LElbow",
        # 7: "LWrist",
        # 8: "RHip",
        # 9: "RKnee",
        # 10: "RAnkle",
        # 11: "LHip",
        # 12: "LKnee",
        # 13: "LAnkle",
        # 14: "REye",
        # 15: "LEye",
        # 16: "REar",
        # 17: "LEar"
    }, [(15, 13), (13, 11), (16, 14),
        (14, 12), (11, 12), (5, 11), (6, 12),
        (5, 6), (7, 5), (8, 6), (9, 7),
        (10, 8), (1, 2), (1, 0), (2, 0),
        (3, 1), (4, 2), (3, 5), (4, 6)]

    # [(0, 1),
    #     (1, 2), (2, 3), (3, 4),
    #     (1, 5), (5, 6), (6, 7),
    #     (1, 8), (8, 9), (9, 10),
    #     (1, 11), (11, 12), (12, 13),
    #     (0, 15), (15, 17), (0, 14), (14, 16)]
)
