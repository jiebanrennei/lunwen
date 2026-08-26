"""测试新加的四个数据集是否能正常加载"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import os.path as osp

# 确保能导入 utils
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from utils import get_dataset

def test_dataset(name):
    print(f"\n{'='*60}")
    print(f"测试数据集: {name}")
    print(f"{'='*60}")
    try:
        datasets = get_dataset(name)
        data = datasets[0]
        print(f"✓ 加载成功!")
        print(f"  节点数: {data.num_nodes}")
        print(f"  边数: {data.num_edges}")
        print(f"  特征维度: {data.x.size(1) if data.x is not None else 'None'}")
        if hasattr(data, 'y') and data.y is not None:
            print(f"  标签数: {data.y.size(0)}")
            if data.y.dim() > 0:
                unique_labels = data.y.unique()
                print(f"  唯一标签数: {unique_labels.size(0)}")
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            print(f"  训练集大小: {data.train_mask.sum().item()}")
            print(f"  验证集大小: {data.val_mask.sum().item()}")
            print(f"  测试集大小: {data.test_mask.sum().item()}")
        return True
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    datasets_to_test = ['com-amazon', 'com-dblp', 'com-youtube', 'twitter']

    results = {}
    for name in datasets_to_test:
        results[name] = test_dataset(name)

    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{name}: {status}")

    all_success = all(results.values())
    sys.exit(0 if all_success else 1)
