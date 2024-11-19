import os
from petrel_client.client import Client

class PatchFolderChecker:
    def __init__(self, paths, client_config='/mnt/petrelfs/yanfang/.petreloss.conf'):
        # 初始化 Client 对象
        self.client = Client(client_config)
        self.paths = paths

    def count_patch_folders(self):
        path_patch_counts = {}
        for base_url in self.paths:
            patch_512_count = self.count_patch_512_in_path(base_url)
            path_patch_counts[base_url] = patch_512_count
            print(f"{base_url} contains {patch_512_count} 'patch_512' folders.")
        return path_patch_counts

    def count_patch_512_in_path(self, base_url):
        """
        递归统计当前路径及其所有子路径中 'patch_512' 文件夹的数量
        """
        try:
            items = self.client.list(base_url)  # 列出当前目录下的所有文件和子目录
        except Exception as e:
            print(f"Error accessing {base_url}: {e}")
            return 0

        patch_512_count = 0

        for item in items:
            item_path = os.path.join(base_url, item)
            
            # 如果当前目录名是 'patch_512'，增加计数
            is_patch_512 = False
            if item.endswith('patch_512'):
                is_patch_512 = True
                patch_512_count += 1

            if item.endswith('patch_256') or item.endswith('region') or item.endswith('.csv'):
                is_patch_512 = True

            # 如果是目录，则递归查找
            if not is_patch_512 and self.is_directory(item_path):
                patch_512_count += self.count_patch_512_in_path(item_path)

        return patch_512_count

    def is_directory(self, path):
        """检查路径是否是目录"""
        try:
            items = self.client.list(path)
            return len(list(items)) > 0  # 如果能列出内容且不为空，说明是目录
        except:
            return False

if __name__ == "__main__":
    # 给定的路径列表
    paths_to_check = [
        'yanfang:s3://yanfang3/TCGA_crop_FFPE/',
        'yanfang:s3://yanfang3/TCGA_crop_frozen/',
        'yanfang:s3://yanfang3/RUJIN_crop/',
        'yanfang:s3://yanfang3/RJ_crop_lymphoma/',
        'yanfang:s3://yanfang3/Digest_ALL_crop_FFPE/',
        'yanfang:s3://yanfang3/Tsinghua_crop/',
        'yanfang:s3://yanfang3/XIJING_crop/',
        'yanfang:s3://yanfang3/IHC_crop_new/',
    ]

    checker = PatchFolderChecker(paths=paths_to_check)
    patch_counts = checker.count_patch_folders()

    print("\nSummary of 'patch_512' folders in each path:")
    for path, count in patch_counts.items():
        print(f"{path}: {count} 'patch_512' folders found")
