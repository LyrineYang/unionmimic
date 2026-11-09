"""Build a source-labeled demo from real rollouts and kinematic previews."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    results_path = ROOT/'work/analysis/geometry-readout-20261109-v1/result.json'
    rows = {r['variant']: r['metrics']['test'] for r in json.loads(results_path.read_text())['results']}
    font = lambda size: ImageFont.truetype(FONT, size)
    scenes = [
        dict(kind='card', duration=4, title='让先验告诉手：物体在哪里',
             subtitle='UnionMimic / SM 学动作，WM-Craftnet 补充几何信息'),
        dict(kind='live', duration=20, source='geometry-live-20261109-v1/geometry-demo.mp4',
             title='真实仿真：SM 控制，冻结先验实时估计物体位置',
             label='历史 SM 策略 · 估计只用于展示 · 未反馈给策略 · 原速'),
        dict(kind='results', duration=8, title='先确认先验带来了什么信息'),
        dict(kind='reference', duration=7,
             source='sharpa-reference-preview-20261109-v2/02_continuous_single_axis.mp4',
             title='拓展目标：单轴参考动作', label='Sharpa 重定向预览 · 运动学播放 · 接触物理尚未验收'),
        dict(kind='reference', duration=7,
             source='sharpa-reference-preview-20261109-v2/03_continuous_multi_axis.mp4',
             title='拓展目标：多轴参考动作', label='Sharpa 重定向预览 · 运动学播放 · 接触物理尚未验收'),
        dict(kind='end', duration=6, title='下一问：在想象中，这份信息能可信多久？'),
    ]
    fps = 25
    writer = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1600x1000', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-crf', '19', '-threads', '4', '-pix_fmt', 'yuv420p',
        str(out/'unionmimic-geometry-story.mp4')], stdin=subprocess.PIPE)
    records = []
    try:
        for si, scene in enumerate(scenes):
            reader = None
            if 'source' in scene:
                path = ROOT/'work/demos'/scene['source']
                reader = imageio_ffmpeg.read_frames(str(path)); meta = next(reader)
                index = -1; frame = None
                records.append(dict(source=str(path), start_s=0, duration_s=scene['duration'],
                                    speed=1, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    label=scene['label']))
            try:
                for f in range(scene['duration']*fps):
                    im = Image.new('RGB', (1600, 1000), '#0c1421'); d = ImageDraw.Draw(im)
                    d.text((42, 16), scene['title'], font=font(32), fill='#eef5ff')
                    if reader:
                        wanted = int(f/fps*meta['fps'])
                        while index < wanted:
                            frame = Image.fromarray(np.frombuffer(next(reader), np.uint8).reshape(meta['size'][1], meta['size'][0], 3))
                            index += 1
                        if scene['kind'] == 'live':
                            im.paste(frame, (80, 63))
                        else:
                            im.paste(frame.resize((1120, 840)), (20, 95))
                            d.text((1180, 260), '动作来自 Wuji', font=font(26), fill='#5eead4')
                            for j, line in enumerate(['重定向到 Sharpa 22', '原速参考播放', '不是学习策略结果', '下一步：物理跟踪']):
                                d.text((1180, 335+j*60), line, font=font(25), fill='#bdcddd')
                        d.text((42, 963), scene['label'], font=font(22), fill='#ffce83')
                    elif scene['kind'] == 'results':
                        d.text((100, 130), '独立仿真种子 / 当前物体中心误差 / 越低越好', font=font(29), fill='#b4c6d9')
                        for j, (key, name) in enumerate([('history', '10 帧观测历史'), ('pretrained', '冻结预训练编码器 + 位置读出'), ('random_502', '同容量随机编码器（3 次：4.07–4.20）'), ('no_depth', '去掉有效深度输入')]):
                            value = rows[key]['mean_mm']; y = 235+j*145
                            d.text((100, y), name, font=font(28), fill='#eef5ff')
                            d.rounded_rectangle((100, y+50, 100+value*245, y+93), radius=8,
                                                fill='#5eead4' if key == 'pretrained' else '#47607a')
                            d.text((130+value*245, y+48), '%.2f mm'%value, font=font(28), fill='#eef5ff')
                        d.text((100, 860), '16 回合、1,934 个采样点；一个行为策略；尚未证明控制收益。', font=font(26), fill='#ffce83')
                    elif scene['kind'] == 'card':
                        d.text((100, 245), 'SM 学动作', font=font(72), fill='#eef5ff')
                        d.text((100, 355), '先验补充物体几何', font=font(72), fill='#5eead4')
                        d.text((100, 540), scene['subtitle'], font=font(32), fill='#b4c6d9')
                        d.text((100, 750), '已测：位置估计    待测：想象中的一致性与控制收益', font=font(31), fill='#ffce83')
                    else:
                        for j, (title, detail) in enumerate([
                            ('已完成：可解释的几何上下文', '冻结编码器，只拟合小型位置读出；同模态随机编码器对照。'),
                            ('已完成：真实 / 想象误差检查', '4 步上下文偏差 2.89 → 2.46 mm；这是离线接口诊断。'),
                            ('下一步：同任务 SM 与 SM + 几何先验', '从随机策略训练，比较姿态误差、掉落和总训练成本。')]):
                            d.text((95, 215+j*205), title, font=font(36), fill='#5eead4')
                            d.text((95, 285+j*205), detail, font=font(29), fill='#eef5ff')
                        d.text((95, 890), '不训练两个完整世界模型；尚无先验改善控制精度的结论。', font=font(28), fill='#ffce83')
                    d.rectangle((0, 996, int(1600*(si+f/(scene['duration']*fps))/len(scenes)), 1000), fill='#5eead4')
                    writer.stdin.write(np.asarray(im).tobytes())
                    if f == fps:
                        im.save(out/('preview-%02d.png'%si))
            finally:
                if reader: reader.close()
    finally:
        writer.stdin.close()
        assert writer.wait(timeout=60) == 0
    (out/'manifest.json').write_text(json.dumps(dict(duration_s=sum(s['duration'] for s in scenes), fps=fps,
        scenes=scenes, clips=records, results=str(results_path), results_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
        control_benefit_claimed=False), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
