import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader

"""
update() 只是把 “一次性计算” 拆成 “多次喂数据”，最终只会生成一个整体的 hash 值，
而非多个 hash 值的拼接 —— 这和字符串拼接是本质不同的。
"""

def get_file_md5_hex(filepath: str):   # 获取文件的的md5的十六进制字符串

    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件(filepath)不存在")
        return

    # 排除文件夹、符号链接、设备文件等
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径(filepath)不是文件")
        return


    """
    通过update()方法增量接收文件二进制数据，
    最终通过hexdigest()生成 MD5 字符串。
    """
    md5_obj = hashlib.md5()

    # 文件分片
    chunk_size = 4096    # 4KB分片，避免文件过大爆内存
    try:
        with open(filepath, 'rb') as f:   # 必须二进制读取
            while chunk := f.read(chunk_size):
                """
                MD5 算法支持 “分片计算”，多次update()分片数据，
                和一次性读取整个文件update()的结果完全一致，既保证准确性，又避免内存溢出。
                """
                md5_obj.update(chunk)
            """
            chunk = f.read(chunk size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            """
            将 MD5 对象的 128 位哈希值（二进制）
            转换为32 位的十六进制字符串（人类可读的形式）
            """
            md5_hex = md5_obj.hexdigest()
            # 返回最终的 MD5 十六进制字符串，供调用方使用
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(dir_path: str, allowed_types: tuple) -> list[str]:
    """
    获取指定目录下指定类型的文件列表（返回绝对路径）
    :param dir_path: 目标文件夹绝对路径
    :param allowed_types: 允许的文件后缀元组，例如 ("txt", "pdf")
    :return: 文件绝对路径列表
    """
    # 1. 严格检查传入的路径是否真的是个文件夹
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        logger.error(f"[listdir_with_allowed_type] 路径不存在或不是文件夹: {dir_path}")
        return []  # 遇到错误必须返回空列表，绝对不能返回 allowed_types ！！！

    file_paths = []

    # 2. 遍历文件夹里面的内容
    try:
        for file_name in os.listdir(dir_path):
            file_abs_path = os.path.join(dir_path, file_name)

            # 3. 如果是文件，且后缀名在允许的列表中（统一转小写比较，防止出现 .PDF 的情况）
            if os.path.isfile(file_abs_path) and file_name.lower().endswith(allowed_types):
                file_paths.append(file_abs_path)

    except Exception as e:
        logger.error(f"[listdir_with_allowed_type] 遍历文件夹失败: {str(e)}")

    return file_paths


def pdf_loader(filepath: str,passwd=None) -> list[Document]:
    return PyPDFLoader(filepath,passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath,encoding="utf-8").load()

