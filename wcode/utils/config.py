import socket
import subprocess
import os
from sys import platform

class TPATH:
    pc_name = socket.gethostname()
    platform = platform
    
    # 检测是否为Windows平台
    IS_WINDOWS = platform.startswith('win') or platform == 'cygwin'
    # 检测是否为macOS平台
    IS_MACOS = platform == 'darwin'
    # 检测是否为Linux平台
    IS_LINUX = platform.startswith('linux')
    
    # 根据平台确定可执行文件扩展名
    EXE_EXTENSION = '.exe' if IS_WINDOWS else ''
    
    # 路径分隔符
    PATH_SEPARATOR = '\\' if IS_WINDOWS else '/'
    
    # 根据平台和主机名设置路径
    if IS_LINUX and pc_name == "Mercury":
        ROSETTA_PATH = "/mnt/e/Program_Files/rosetta.source.release-371/"
        SCHRODINGER_PATH = "/mnt/e/Program_Files/Schrodinger_202402/"
        RMSD_SCRIPT = r"E:\\Program_Files\\Schrodinger_202402\\mmshare-v6.6\\python\\common\\rmsd.py"

    elif IS_LINUX and pc_name == "Venus":
        ROSETTA_PATH = "/home/fluoxetine/rosetta.source.release-371/"
        SCHRODINGER_PATH = "/mnt/c/Program\ Files/Schrodinger2024-2/"

    elif IS_LINUX and pc_name == "Jupiter":
        ROSETTA_PATH = "/root/rosetta.source.release-371"
        SCHRODINGER_PATH = None  # 需要根据实际情况设置
    
    elif IS_MACOS and pc_name == "wangdingyandeMacBook-Pro.local":
        SCHRODINGER_PATH = "/opt/schrodinger/suites2025-2"
        ROSETTA_PATH = None  # 需要根据实际情况设置
    
    else:
        # 默认配置，可以根据需要修改
        ROSETTA_PATH = None
        SCHRODINGER_PATH = None
        RMSD_SCRIPT = None
    
    @classmethod
    def get_executable_path(cls, base_path, program_name):
        """
        根据平台自动生成可执行文件路径
        
        Args:
            base_path: 基础路径
            program_name: 程序名称（不含扩展名）
            
        Returns:
            完整的可执行文件路径
        """
        if not base_path:
            return None
        
        # 确保路径以正确的分隔符结尾
        if not base_path.endswith(cls.PATH_SEPARATOR):
            base_path += cls.PATH_SEPARATOR
        
        return base_path + program_name + cls.EXE_EXTENSION
    
    @classmethod
    def get_rosetta_executable(cls, program_name):
        """获取Rosetta可执行文件路径"""
        return cls.get_executable_path(cls.ROSETTA_PATH, program_name)
    
    @classmethod
    def get_schrodinger_executable(cls, program_name):
        """获取Schrodinger可执行文件路径"""
        return cls.get_executable_path(cls.SCHRODINGER_PATH, program_name)
    
    # Rosetta-related (使用新的路径生成方法)
    @property
    def SIMPLEPEP(self):
        return self.get_rosetta_executable("main/source/bin/simple_cycpep_predict.mpi.linuxgccrelease")
    
    @property
    def SILENT_SPLIT(self):
        return self.get_rosetta_executable("main/source/bin/extract_pdbs.mpi.linuxgccrelease")
    
    # Schrodinger-related (使用新的路径生成方法)
    @property
    def SCHRODINGER_RUN(self):
        return self.get_schrodinger_executable("run")
    
    @property
    def GLIDE(self):
        return self.get_schrodinger_executable("glide")
    
    @property
    def LIGPREP(self):
        return self.get_schrodinger_executable("ligprep")
    
    @property
    def STRUCTCONVERT(self):
        return self.get_schrodinger_executable("utilities/structconvert")
    
    @property
    def PROTEINPREP(self):
        return self.get_schrodinger_executable("utilities/prepwizard")


def convert_wsl_to_windows_path(wsl_path, arg='-w'):
    """
    使用wslpath命令将WSL路径转换为Windows路径
    
    Args:
        wsl_path: WSL路径
        arg: wslpath参数，'-w'表示转换为Windows路径
        
    Returns:
        转换后的Windows路径
    """
    try:
        result = subprocess.run(['wslpath', arg, wsl_path], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        if result.returncode == 0:
            windows_path = result.stdout.strip()
            # 转义反斜杠
            windows_path = windows_path.replace('\\', '\\\\')
            return windows_path
        else:
            print(f"wslpath转换失败: {result.stderr}")
            return wsl_path
    except FileNotFoundError:
        print("wslpath命令未找到，可能不在WSL环境中")
        return wsl_path
    except Exception as e:
        print(f"路径转换出错: {e}")
        return wsl_path


def get_platform_info():
    """
    获取当前平台信息
    
    Returns:
        包含平台信息的字典
    """
    return {
        'platform': platform,
        'hostname': socket.gethostname(),
        'is_windows': TPATH.IS_WINDOWS,
        'is_macos': TPATH.IS_MACOS,
        'is_linux': TPATH.IS_LINUX,
        'exe_extension': TPATH.EXE_EXTENSION,
        'path_separator': TPATH.PATH_SEPARATOR
    }


if __name__ == '__main__':
    # 测试平台信息
    platform_info = get_platform_info()
    print("平台信息:")
    for key, value in platform_info.items():
        print(f"  {key}: {value}")
    
    # 测试路径转换
    if TPATH.IS_LINUX:
        wsl_path = '/tmp'
        windows_path = convert_wsl_to_windows_path(wsl_path)
        print(f"\nWSL路径转换测试:")
        print(f"  WSL路径: {wsl_path}")
        print(f"  Windows路径: {windows_path}")
    
    # 测试可执行文件路径生成
    config = TPATH()
    print(f"\n可执行文件路径测试:")
    print(f"  SIMPLEPEP: {config.SIMPLEPEP}")
    print(f"  GLIDE: {config.GLIDE}")
    print(f"  LIGPREP: {config.LIGPREP}")


