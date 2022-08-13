from skeleton_tools.skeleton_visualization.draw_utils import draw_bbox, blur_area


class PyfeatVisualizer:
    def __init__(self, blur_face):
        self.blur_face = blur_face

    def draw_facebox(self, frame, faces):
        for _, face in faces.iterrows():
            bbox = face[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']].astype(int).values.reshape(2, 2)
            if self.blur_face:
                frame = blur_area(frame, bbox[0], bbox[1].max())
            color = (255, 0, 0) if face['is_child'] else (0, 0, 255)
            draw_bbox(frame, bbox, color)
        return frame