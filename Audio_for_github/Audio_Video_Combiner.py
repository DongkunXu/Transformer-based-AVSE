import os
from moviepy import VideoFileClip, AudioFileClip


def combine_audio_video(input_video_path, input_audio_path, output_path):
    """
    将音频文件与无声视频文件合并，生成新的视频文件

    参数：
    input_video_path (str): 输入视频文件路径
    input_audio_path (str): 输入音频文件路径
    output_path (str): 输出视频文件路径
    """
    try:
        # 加载视频并移除原有音轨
        video_clip = VideoFileClip(input_video_path).without_audio()

        # 加载音频
        audio_clip = AudioFileClip(input_audio_path)

        # 设置视频的音频轨
        final_clip = video_clip.with_audio(audio_clip)

        # 保持原始视频参数
        final_clip.write_videofile(
            output_path,
            codec='libx264',  # 保持H.264编码
            audio_codec='aac',  # 保持AAC音频编码
            fps=video_clip.fps,  # 保持原始帧率
            threads=4,  # 使用多线程加速处理
        )

        # 显式释放资源
        video_clip.close()
        audio_clip.close()
        final_clip.close()

        print(f"成功生成：{output_path}")

    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")
        raise


def main():
    # 参数配置（集中管理）
    config = {
        "base_dir": r"E:\School_Work\PhD_1\Transformer-AVSE-V5\Audio_for_github",
        "silent_video": "S00047_silent.mp4",
        "audio_files": [
            ("mixed.wav", "mixed.mp4"),
            ("enhanced.wav", "enhanced.mp4")
        ]
    }

    # 获取完整路径
    silent_video_path = os.path.join(config["base_dir"], config["silent_video"])

    # 处理所有音频文件
    for audio_file, output_file in config["audio_files"]:
        input_audio = os.path.join(config["base_dir"], audio_file)
        output_path = os.path.join(config["base_dir"], output_file)

        print(f"正在处理：{audio_file} -> {output_file}")
        combine_audio_video(silent_video_path, input_audio, output_path)


if __name__ == "__main__":
    main()