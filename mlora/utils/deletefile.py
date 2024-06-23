import os
import shutil
import os.path

def delete_files_in_folder(folder_path):
    try:
        # 遍历目录中的每个项目
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):  # 如果是文件，则删除
                os.remove(item_path)
            elif os.path.isdir(item_path):  # 如果是目录，则递归删除
                delete_files_in_folder(item_path)

        # 尝试删除空目录
        try:
            os.rmdir(folder_path)
            print(f"Successfully deleted all contents of {folder_path} and the folder itself.")
        except OSError as e:
            # 如果目录不为空（或有其他问题），则捕获异常并打印消息
            print(f"Failed to delete the empty folder {folder_path}. Reason: {e}")

    except Exception as e:
        # 如果在遍历或删除文件/目录时发生其他异常，则捕获并打印消息
        print(f"Failed to delete files/folders in {folder_path}. Reason: {e}")