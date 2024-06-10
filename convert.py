import os
from moviepy.editor import VideoFileClip

def convert_fps(input_video_path, output_video_path, target_fps):
    # 检查输入视频文件是否存在
    if not os.path.isfile(input_video_path):
        print(f"Error: The file {input_video_path} could not be found!")
        return
    # 加载输入视频
    clip = VideoFileClip(input_video_path)
    # 将视频帧率转换为目标帧率
    new_clip = clip.set_fps(target_fps)
    # 确保输出目录存在
    os.makedirs(output_video_path, exist_ok=True)
    # 获取视频文件名并去除扩展名
    base_name = os.path.basename(input_video_path).split('.')[0]
    # 保存输出视频
    new_clip.write_videofile(os.path.join(output_video_path, f'{base_name}_{target_fps}fps.mp4'), fps=target_fps)

# 参数配置区--------------------------------------------------
video_annotation_pairs = [
    #('video/1.mp4', 'JSON/1.json'),
    ('video/2.mp4', 'JSON/2.json'),
    ('video/3.mp4', 'JSON/3.json'),
    ('video/4.mp4', 'JSON/4.json'),
    #('video/5.mp4', 'JSON/5.json'),
    ('video/6.mp4', 'JSON/5.json'),
    ('video/7.mp4', 'JSON/5.json'),
    ('video/53101.mp4', 'JSON/53101.json'),
    ('video/53102.mp4', 'JSON/53102.json'),
    ('video/53103.mp4', 'JSON/53103.json'),
    ('video/53104.mp4', 'JSON/53104.json'),
    ('video/60701.mp4', 'JSON/60701.json'),
    ('video/60702.mp4', 'JSON/60702.json'),
    ('video/60703.mp4', 'JSON/60703.json'),
    ('video/60704.mp4', 'JSON/60704.json'),
    ('video/60705.mp4', 'JSON/60705.json'),
    ('video/60706.mp4', 'JSON/60706.json'),
    ('video/60707.mp4', 'JSON/60707.json'),
    ('video/60708.mp4', 'JSON/60708.json'),
    ('video/60709.mp4', 'JSON/60709.json'),
    ('video/60710.mp4', 'JSON/60710.json'),
    ('video/60711.mp4', 'JSON/60711.json'),
    ('video/60712.mp4', 'JSON/60712.json'),
    #('video/60801.mp4', 'JSON/60801.json'),
    ('video/60802.mp4', 'JSON/60802.json'),
    #('video/60803.mp4', 'JSON/60803.json'),
    ('video/60804.mp4', 'JSON/60804.json'),
    ('video/60805.mp4', 'JSON/60805.json'),
    ('video/60806.mp4', 'JSON/60806.json'),
    ('video/60807.mp4', 'JSON/60807.json')
    # (视频文件, 标注文件)
]
target_fps = 5
output_video_path = 'output_video'
# ------------------------------------------------------------

for input_video_path, _ in video_annotation_pairs:
    convert_fps(input_video_path, output_video_path, target_fps)
