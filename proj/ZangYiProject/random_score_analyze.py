import numpy as np

# 定义20种常见氨基酸
AMINO_ACIDS = 'CDEFGHIKLMNPQRSTVWYA'


def analyze_scores(file_name):
    # 读取已保存的序列和分数
    data = np.load(file_name, allow_pickle=True)
    sequences = data['sequences']
    scores = data['scores']

    analysis_results = []

    # 对第二到第七位进行分析
    for pos in range(1, 8):
        for aa in AMINO_ACIDS:
            filtered_scores = [score for seq, score in zip(sequences, scores) if seq[pos] == aa]
            if filtered_scores:
                mean_score = np.mean(filtered_scores)
                analysis_results.append((f'Position {pos + 1} Amino Acid {aa}', mean_score))

    # 按均值从大到小排序
    analysis_results.sort(key=lambda x: x[1], reverse=True)

    return analysis_results


def main():
    file_name = '/mnt/c/tmp/peptide_scores.npz'
    results = analyze_scores(file_name)

    # 打印结果
    for result in results:
        print(f'{result[0]}: {result[1]:.4f}')


if __name__ == '__main__':
    main()