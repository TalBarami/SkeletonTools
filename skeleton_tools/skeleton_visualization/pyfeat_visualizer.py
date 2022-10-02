from skeleton_tools.skeleton_visualization.draw_utils import draw_bbox, blur_area
from skeleton_tools.skeleton_visualization.skeleton_visualizer import SkeletonVisualizer
from skeleton_tools.utils.tools import get_video_properties


class PyfeatVisualizer(SkeletonVisualizer):
    def __init__(self, graph_layout, display_pid=False, display_bbox=False, denormalize=False, decentralize=False, blur_face=False, show_confidence=False):
        super().__init__(graph_layout, display_pid, display_bbox, denormalize, decentralize, blur_face, show_confidence)

    # def draw_facebox(self, frame, faces):
    #     for _, face in faces.iterrows():
    #         bbox = face[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']].astype(int).values.reshape(2, 2)
    #         if self.blur_face:
    #             frame = blur_area(frame, bbox[0], bbox[1].max())
    #         color = (255, 0, 0) if face['is_child'] else (0, 0, 255)
    #         draw_bbox(frame, bbox, color)
    #     return frame

    def get_video_info(self, video_path, pyfeat_data):
        raise NotImplementedError('TBI')
        # resolution, fps, frame_count, length = get_video_properties(video_path)
        # return fps, length, resolution, kp, c, pids, child_ids, detections, child_bbox, adjust