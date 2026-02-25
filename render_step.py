import os
import time
from tqdm import tqdm
from PIL import Image
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Display.SimpleGui import init_display
import math
import subprocess
import sys


class STEPRenderer:
    def __init__(self, view_size=100):
        """初始化离屏显示器"""
        # Set up virtual display for headless environment
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['DISPLAY'] = ':99'
        
        try:
            self.display, self.start_display, self.add_menu, self.add_function = init_display(
                size=(view_size, view_size), 
                backend_str='qt-pyqt5'
            )
            self.display.Context.SetDisplayMode(1, True)
            self.view_size = view_size
        except Exception as e:
            raise RuntimeError(f"Failed to initialize display: {e}")

    def load_step(self, step_file_path):
        """加载STEP文件"""
        try:
            if not os.path.exists(step_file_path):
                raise FileNotFoundError(f"找不到STEP文件: {step_file_path}")
            self.display.Context.RemoveAll(True)
            self.shape = read_step_file(step_file_path)
            self.display.DisplayShape(self.shape, update=True)
        except Exception as e:
            raise RuntimeError(f"加载文件失败: {step_file_path}, 错误: {e}")


    def calculate_model_bounds(self):
        """计算模型的边界框并返回缩放因子"""
        bbox = Bnd_Box()
        brepbndlib_Add(self.shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return max(xmax - xmin, ymax - ymin, zmax - zmin)

    def set_view(self, dx, dy, dz):
        """设置视图方向"""
        self.display.View.SetProj(dx, dy, dz)
        self.display.FitAll()
        scale_factor = self.calculate_model_bounds()
        if scale_factor > 0:
            zoom_factor = 1.2 / scale_factor
            self.display.View.SetZoom(zoom_factor)
        self.display.View.Camera().SetProjectionType(1)
        self.display.View.TriedronErase()
        # 强制刷新视图
        self.display.View.Update()
        time.sleep(0.02)  # 延时确保渲染完成

    def capture_view(self):
        """捕获当前视图"""
        self.display.FitAll()
        self.display.View.SetZoom(0.8)
        temp_file = './temp_view.jpeg'
        self.display.View.Dump(temp_file)
        img = Image.open(temp_file)
        width, height = img.size
        if width > height:
            offset = (width - height) // 2
            crop_box = (offset, 0, offset + height, height)
        else:
            offset = (height - width) // 2
            crop_box = (0, offset, width, offset + width)
        img = img.crop(crop_box).resize((self.view_size, self.view_size), Image.LANCZOS)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return img

    def render_combined_view(self, step_file_path, output_path):
        """渲染并组合9个视图"""
        self.load_step(step_file_path)
        spherical_coords = [
            (0, 90), (0, 0), (90, 90),
            (45, 45), (45, 135), (225, 45),
            (225, 135), (0, 60), (0, 120)
        ]
        view_images = []
        for coords in spherical_coords:
            dx, dy, dz = self.spherical_to_cartesian(*coords)
            self.set_view(dx, dy, dz)
            img = self.capture_view()
            view_images.append(img)
        combined_img = Image.new('RGB', (self.view_size * 3, self.view_size * 3), 'white')
        for i, img in enumerate(view_images):
            row = i // 3
            col = i % 3
            combined_img.paste(img, (col * self.view_size, row * self.view_size))
        combined_img.save(output_path, format="JPEG")

    def spherical_to_cartesian(self, theta, phi):
        """球面坐标转笛卡尔坐标"""
        theta_rad = math.radians(theta)
        phi_rad = math.radians(phi)
        x = math.sin(phi_rad) * math.cos(theta_rad)
        y = math.sin(phi_rad) * math.sin(theta_rad)
        z = math.cos(phi_rad)
        return (x, y, z)


def run_rendering():
    """主渲染函数"""
    base_path = '/home/group/cad_codebased/data/abccad/step_500-1000/abc_0010_step_v00_500-1000'
    output_dir = '/home/group/cad_codebased/data/abccad/step_500-1000_image/abc_0010_step_v00_500-1000_image'
    os.makedirs(output_dir, exist_ok=True)
    view_size = 100
    
    try:
        renderer = STEPRenderer(view_size=view_size)
        
        # 动态获取子目录和 .step 文件
        step_files = []
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):  # 确保是目录
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.step'):  # 筛选出 .step 文件
                        step_files.append(os.path.join(folder_path, file_name))

        print(f"Found {len(step_files)} STEP files to process")
        
        # 遍历并处理所有文件，显示进度条
        for step_file in tqdm(step_files, desc="处理STEP文件", unit="文件"):
            output_file = os.path.join(output_dir, f"{os.path.basename(step_file).rsplit('.', 1)[0]}.jpeg")
            try:
                renderer.render_combined_view(step_file, output_file)
            except Exception as e:
                print(f"处理文件时出错: {step_file}, 错误: {e}")
                
    except Exception as e:
        print(f"初始化渲染器失败: {e}")
        return 1
    
    return 0


def main():
    """主函数 - 检测是否在虚拟显示环境中运行"""
    # 检查是否已经在虚拟显示环境中
    if os.environ.get('DISPLAY') == ':99' or os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        # 已经在虚拟显示环境中，直接运行
        return run_rendering()
    
    # 检查是否可以使用xvfb-run
    try:
        subprocess.run(['which', 'xvfb-run'], check=True, capture_output=True)
        print("Using xvfb-run for virtual display...")
        # 使用xvfb-run重新启动脚本
        cmd = ['xvfb-run', '-a', '-s', '-screen 0 1024x768x24', 
               sys.executable, __file__]
        result = subprocess.run(cmd)
        return result.returncode
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("xvfb-run not available, attempting direct execution...")
        return run_rendering()


if __name__ == "__main__":
    sys.exit(main())