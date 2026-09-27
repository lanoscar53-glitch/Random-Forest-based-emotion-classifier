# -*- coding: utf-8 -*-
"""
============================================================================
 weibo_senti_100k 微博情感二分类 —— 纯 NumPy 手写 TF-IDF + 随机森林（精简版）
============================================================================
【核心算法（全部手写，只用 NumPy）】
    1) TF-IDF：tfidf(t,d) = (1+log(tf)) × (log((1+N)/(1+df))+1)，再 L2 归一化；
    2) 随机森林：Bootstrap 抽样 + CART 决策树（基尼不纯度 Gini=2p(1-p)）
                 + 节点级随机特征子集 + 多树投票平均。
    仅 jieba 用于中文分词（预处理，非算法部分）。

【运行】python weibo_senti_rf_numpy.py
============================================================================
"""
import re
import time

import jieba
import numpy as np
import pandas as pd

# ------------------------- 配置 -------------------------
DATA_PATH = r"D:\datasets\weibo_senti_100k\weibo_senti_100k.csv"
RANDOM_STATE = 42
N_SAMPLE = 20000     # 抽样条数（密集矩阵内存考虑；改 0 用全量，需更大内存）
N_TREES = 30         # 树的数量
MAX_DEPTH = 10       # 树最大深度
N_FEATURES = 3000    # TF-IDF 词表大小

# ------------------------- 文本预处理 -------------------------
def tokenize(text: str) -> str:
    """清洗 + jieba 分词：去链接/@，[表情]转文字（强情感特征，保留）。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|@[\w\u4e00-\u9fa5\-]+", " ", text)  # 去 URL 和 @
    text = re.sub(r"\[([^\[\]]{1,6})\]", r"\1", text)                # [哈哈] -> 哈哈
    return " ".join(w for w in jieba.lcut(text) if w.strip())

# ============================================================================
# 核心算法 1：纯 NumPy 手写 TF-IDF
# ============================================================================
class NumpyTfidfVectorizer:
    """
    tfidf(t, d) = tf(t, d) × idf(t)
      - tf(t,d) = 1 + log(词 t 在文档 d 中的次数)   （sublinear，抑制高频词霸榜）
      - idf(t)  = log((1+文档总数N) / (1+含词t的文档数df)) + 1
                  词越"遍地都是"权重越低，越"稀有"权重越高
      - 最后每个文档向量做 L2 归一化，消除文本长短差异
    """

    def __init__(self, max_features=3000, min_df=3):
        self.max_features = max_features  # 词表大小上限
        self.min_df = min_df              # 至少出现在多少篇文档中才收入词表
        self.vocab_ = {}                  # 词 -> 列号
        self.idf_ = None                  # 每个词的 idf 值

    def fit(self, texts):
        """在【训练集】上学习词表和 IDF（测试集只 transform，防信息泄露）。"""
        df_counter, tf_counter = {}, {}
        for text in texts:
            words = text.split()
            for w in words:
                tf_counter[w] = tf_counter.get(w, 0) + 1
            for w in set(words):                      # 每篇文档对 df 只贡献 1 次
                df_counter[w] = df_counter.get(w, 0) + 1
        # 过滤低频词，按总词频取前 max_features 个建词表
        candidates = [(w, c) for w, c in tf_counter.items()
                      if df_counter[w] >= self.min_df]
        candidates.sort(key=lambda x: -x[1])
        self.vocab_ = {w: i for i, (w, _) in enumerate(candidates[:self.max_features])}
        # idf = log((1+N)/(1+df)) + 1（平滑公式）
        n_docs = len(texts)
        self.idf_ = np.array([np.log((1 + n_docs) / (1 + df_counter[w])) + 1
                              for w in self.vocab_], dtype=np.float32)
        return self

    def transform(self, texts) -> np.ndarray:
        """词串列表 -> TF-IDF 矩阵 [n_docs, n_features]（float32）。"""
        X = np.zeros((len(texts), len(self.vocab_)), dtype=np.float32)
        for i, text in enumerate(texts):
            col_ids = [self.vocab_[w] for w in text.split() if w in self.vocab_]
            if not col_ids:
                continue
            tf = np.bincount(col_ids, minlength=len(self.vocab_)).astype(np.float32)
            log_tf = np.zeros_like(tf)
            np.log(tf, out=log_tf, where=tf > 0)      # 只对出现过的词取 log
            X[i] = np.where(tf > 0, 1.0 + log_tf, 0.0) * self.idf_   # tf × idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)             # L2 归一化
        return np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

# ============================================================================
# 核心算法 2：纯 NumPy 手写 CART 决策树（随机森林的基学习器）
# ============================================================================
class NumpyDecisionTree:
    """
    二分类 CART 树：每个节点选使【加权基尼不纯度】最小的切分。
      - Gini(组) = 2p(1-p)，p 为组内正样本比例；切分越"纯"，加权 Gini 越小
      - 候选切分点：该特征非零值的 5 个分位数（近似最优，速度快）
      - 停止条件：深度上限 / 样本太少 / 节点已纯 / 无合法切分
      - 叶节点的值 = 叶内正样本比例，即这棵树的 P(正向)
      - 每个节点只随机看 max_features 个特征（随机森林的第二重随机）
    """

    def __init__(self, max_depth=10, min_samples_leaf=20, max_features=50, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()
        self.root = None

    def _best_split(self, X, y):
        n, p = X.shape
        feat_ids = self.rng.choice(p, size=min(self.max_features, p), replace=False)
        best = None                                          # (gini, 特征号, 阈值)
        for f in feat_ids:
            col = X[:, f]
            nz = col[col > 0]                                # TF-IDF 大多是 0，只看非零
            if nz.size < 2 * self.min_samples_leaf:
                continue
            for t in np.unique(np.quantile(nz, [0.1, 0.3, 0.5, 0.7, 0.9])):
                left = col <= t
                nl, nr = int(left.sum()), n - int(left.sum())
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                pl, pr = y[left].mean(), y[~left].mean()     # 左右组正样本比例
                gini = (nl * 2 * pl * (1 - pl) + nr * 2 * pr * (1 - pr)) / n
                if best is None or gini < best[0]:
                    best = (gini, f, t)
        return best

    def _build(self, X, y, depth):
        prob = float(y.mean())                               # 当前节点 P(正向)
        if (depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf
                or prob == 0.0 or prob == 1.0):
            return {"leaf": True, "prob": prob}              # 停止 -> 叶节点
        best = self._best_split(X, y)
        if best is None:
            return {"leaf": True, "prob": prob}
        _, f, t = best
        left = X[:, f] <= t
        return {"leaf": False, "feat": f, "thr": t,
                "left": self._build(X[left], y[left], depth + 1),
                "right": self._build(X[~left], y[~left], depth + 1)}

    def fit(self, X, y):
        self.root = self._build(X, y.astype(np.float32), 0)
        return self

    def predict_one(self, x) -> float:
        """单样本预测：从根走到叶，返回叶节点的 P(正向)。"""
        node = self.root
        while not node["leaf"]:
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return node["prob"]

# ============================================================================
# 核心算法 3：纯 NumPy 手写随机森林
# ============================================================================
class NumpyRandomForest:
    """
    随机森林 = N 棵树 + 双重随机 + 投票。
      - 第一重随机：每棵树对训练集 Bootstrap 有放回抽样（约 63.2% 唯一样本）
      - 第二重随机：节点分裂时随机抽 sqrt(总特征数) 个特征竞选
      - 森林概率 = 所有树叶节点概率的平均，>= 0.5 判正向
      - 树与树"错得不一样"，投票时错误相互抵消（集成降方差）
    """

    def __init__(self, n_trees=30, max_depth=10, min_samples_leaf=20,
                 max_features=50, random_state=42):
        self.n_trees, self.max_depth = n_trees, max_depth
        self.min_samples_leaf, self.max_features = min_samples_leaf, max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        n = len(y)
        for i in range(self.n_trees):
            rng = np.random.default_rng(self.random_state + i)   # 每棵树独立种子
            idx = rng.integers(0, n, n)                          # Bootstrap 抽样
            tree = NumpyDecisionTree(self.max_depth, self.min_samples_leaf,
                                     self.max_features, rng)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            print(f"      第 {i + 1}/{self.n_trees} 棵树完成", end="\r")
        print()
        return self

    def predict_proba(self, X) -> np.ndarray:
        """森林的 P(正向) = 所有树预测概率的平均（即投票比例）。"""
        votes = np.array([[t.predict_one(x) for x in X] for t in self.trees])
        return votes.mean(axis=0)

# ============================================================================
# 随机抽样判别示例
# ============================================================================
def random_demo(forest, X_test, y_test, raw_texts, n=50):
    """从测试集随机抽 n 条逐条判别（每次运行样本不同），并统计本次示例的准确率。"""
    rng = np.random.default_rng()                          # 不固定种子：每次抽样不同
    idxs = rng.choice(len(y_test), size=min(n, len(y_test)), replace=False)
    print(f"\n>>> 随机抽样 {len(idxs)} 条判别示例（每次运行样本不同）:")
    print(f"    {'序号':<5}{'真实':<5}{'预测':<5}{'置信度':<8}{'投票(负:正)':<13}原文")
    print("    " + "-" * 78)
    n_correct = 0
    for rank, i in enumerate(idxs, 1):
        votes = np.array([1 if t.predict_one(X_test[i]) >= 0.5 else 0
                          for t in forest.trees])          # 每棵树的硬投票
        prob_pos = votes.mean()                            # 正向得票率即森林概率
        pred = int(prob_pos >= 0.5)
        conf = max(prob_pos, 1 - prob_pos)                 # 置信度 = 多数派比例
        ok = pred == y_test[i]
        n_correct += ok
        print(f"    {rank:<5}{'负向' if y_test[i] == 0 else '正向':<5}"
              f"{'负向' if pred == 0 else '正向':<5}{conf:<8.2f}"
              f"{f'{len(votes) - votes.sum()}:{votes.sum()}':<13}"
              f"{raw_texts[i][:28]} {'✓' if ok else '✗'}")
    print("    " + "-" * 78)
    print(f">>> 示例准确率: {n_correct}/{len(idxs)} = {n_correct / len(idxs):.1%}")

# ============================================================================
# 主流程
# ============================================================================
def main():
    # [1/4] 读取数据 + 清洗分词 + 分层抽样
    print(f"[1/4] 读取数据并预处理: {DATA_PATH}")
    t0 = time.time()
    df = pd.read_csv(DATA_PATH).dropna(subset=["label", "review"])
    df["label"] = df["label"].astype(int)
    if 0 < N_SAMPLE < len(df):                               # 分层抽样（正负各半）
        df = df.groupby("label").sample(
            n=N_SAMPLE // 2, random_state=RANDOM_STATE).reset_index(drop=True)
    df["tokens"] = df["review"].astype(str).map(tokenize)
    df = df[df["tokens"].str.len() > 0].reset_index(drop=True)
    print(f"      样本数: {len(df)}, 耗时 {time.time() - t0:.1f}s")

    # [2/4] 手写 8:2 分层划分（每类各自打乱后按比例切开）
    print("[2/4] 划分训练集/测试集（80% : 20%，分层）")
    rng = np.random.default_rng(RANDOM_STATE)
    train_idx, test_idx = [], []
    for label in (0, 1):
        idx = np.where(df["label"].values == label)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * 0.8)
        train_idx.append(idx[:cut])
        test_idx.append(idx[cut:])
    train_idx, test_idx = np.concatenate(train_idx), np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    print(f"      训练集: {len(train_idx)} 条, 测试集: {len(test_idx)} 条")

    # [3/4] 手写 TF-IDF（只在训练集上 fit）
    print(f"[3/4] 手写 TF-IDF 向量化（词表 {N_FEATURES} 维）")
    t0 = time.time()
    vec = NumpyTfidfVectorizer(max_features=N_FEATURES, min_df=3)
    X_train = vec.fit_transform(df["tokens"].values[train_idx])
    X_test = vec.transform(df["tokens"].values[test_idx])
    y_train, y_test = df["label"].values[train_idx], df["label"].values[test_idx]
    print(f"      矩阵: 训练 {X_train.shape}, 测试 {X_test.shape}, 耗时 {time.time() - t0:.1f}s")

    # [4/4] 手写随机森林训练 + 评估
    print(f"[4/4] 手写随机森林训练（{N_TREES} 棵树, 深度 {MAX_DEPTH}）")
    t0 = time.time()
    forest = NumpyRandomForest(n_trees=N_TREES, max_depth=MAX_DEPTH,
                               max_features=max(1, int(np.sqrt(N_FEATURES))),
                               random_state=RANDOM_STATE)
    forest.fit(X_train, y_train)
    print(f"      训练耗时 {time.time() - t0:.1f}s")

    prob_pos = forest.predict_proba(X_test)              # 森林的 P(正向)
    y_pred = (prob_pos >= 0.5).astype(int)               # 0.5 为界判正/负
    acc = float((y_pred == y_test).mean())
    # 混淆矩阵 [[TN, FP], [FN, TP]]
    cm = np.array([[int(((y_test == 0) & (y_pred == 0)).sum()),
                    int(((y_test == 0) & (y_pred == 1)).sum())],
                   [int(((y_test == 1) & (y_pred == 0)).sum()),
                    int(((y_test == 1) & (y_pred == 1)).sum())]])
    print(f"\n>>> 二分类准确率 Accuracy = {acc:.4f}")
    print(">>> 混淆矩阵（行=真实, 列=预测）:")
    print(cm)

    # 随机抽样 50 条判别示例（含每棵树投票数与示例准确率）
    raw_test = [t.replace(" ", "") for t in df["tokens"].values[test_idx]]
    random_demo(forest, X_test, y_test, raw_test, n=50)

if __name__ == "__main__":
    main()
