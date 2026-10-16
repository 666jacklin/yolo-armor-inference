一，## 项目功能

本项目基于 YOLO26n 装甲板模型 `best.pt`，对自行拍摄的装甲板视频进行推理。

脚本 `infer.py` 主要完成以下功能：

- 加载模型权重；
- 读取输入视频并逐帧推理；
- 检测画面中的装甲板目标；
- 在检测结果中显示检测框、类别名称、置信度；
- 显示该目标的 4 个关键点；
- 保存推理结果视频。

当模型权重为纯检测模型时，4 个关键点由检测框的左上、右上、右下、左下四个角点计算得到，并在可视化中标注序号，使其清晰对应到所属目标。

二，## 运行环境

- Python 版本：Python 3.10
- 主要依赖：ultralytics、opencv-python、numpy、torch
- 推理设备：CPU

本项目当前在 CPU 环境下运行，未使用 GPU。若使用 GPU，可将推理参数中的 `device` 改为 `0`，并安装对应 CUDA 版本的 PyTorch。

三## 模型权重准备

模型权重 `best.pt` 不提交到 Git 仓库中。

运行前需要将 `best.pt` 放入项目根目录：

```text
yolo-armor-inference/
├── best.pt        # 自行准备，不提交到 Git
├── infer.py
├── requirements.txt
├── README.md
└── .gitignore

四，安装依赖
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

运行推理
python infer.py --weights best.pt --input VID20261015151632.mp4 --output result.mp4

结果保存位置
result.mp4

五，在水建楼d402教室拍摄，时长19秒。

七，实际识别表现应以真实运行结果为准。运行
后请观察以下情况:
是否检测到装甲板目标；
检测框是否稳定贴合目标；
类别名称是否正确；
置信度数值是否合理；
4个关键点是否准确对应到目标位置；
是否出现漏检、误检、类别错误或关键点
位置偏差。
若出现全片没有有效检测，应依次检查：
1. best.pt 是否已正确放置在项目根目录；
2.输入视频路径是否正确；
3.模型类别是否与视频目标匹配；
4.置信度阈值--conf是否过高;
5.依赖环境是否安装完整。
