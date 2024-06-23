import os
import shutil
import os.path

def delete_files_in_folder(folder_path):
    try:
        #Iterate through each file in the directory
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):  #If it's a file, delete it
                os.remove(item_path)
            elif os.path.isdir(item_path):  #If it's a directory, it's deleted recursively
                delete_files_in_folder(item_path)

        #Delete empty directories
        try:
            os.rmdir(folder_path)
            print(f"Successfully deleted all contents of {folder_path} and the folder itself.")
        except OSError as e:
            #If the directory is not empty (or has some other problem), the exception is caught and the message is printed
            print(f"Failed to delete the empty folder {folder_path}. Reason: {e}")

    except Exception as e:
        #If other exceptions occur while traversing or deleting files/directories, the message is captured and printed
        print(f"Failed to delete files/folders in {folder_path}. Reason: {e}")
