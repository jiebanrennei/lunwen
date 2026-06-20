"""
对抗模式库加载器和意图编码器
Adversarial Pattern Library Loader and Intent Encoder

用于加载对抗模式库并生成意图向量
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import os

# 尝试导入sentence_transformers，如果没有安装则提供提示
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence_transformers not installed. Run: pip install sentence-transformers")


class AdversarialPatternLibrary:
    """
    对抗模式库管理器

    功能：
    1. 加载JSON格式的对抗模式库
    2. 预计算模式嵌入向量
    3. 根据查询匹配相关模式
    4. 支持按类别检索模式
    """

    def __init__(self,
                 library_path: str = None,
                 encoder_name: str = 'BAAI/bge-large-zh-v1.5',
                 device: str = None):
        """
        初始化对抗模式库

        Args:
            library_path: JSON库文件路径，默认使用同目录下的adversarial_pattern_library.json
            encoder_name: 文本编码器名称
            device: 计算设备 ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模式库
        if library_path is None:
            library_path = os.path.join(os.path.dirname(__file__), 'adversarial_pattern_library.json')

        self.library_data = self._load_library(library_path)
        self.patterns = self.library_data['patterns']
        self.categories = self.library_data['categories']
        self.metadata = self.library_data['metadata']

        # 初始化编码器
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading encoder: {encoder_name}")
            self.encoder = SentenceTransformer(encoder_name)
            self.encoder_dim = self.encoder.get_sentence_embedding_dimension()

            # 预计算模式嵌入
            self.pattern_embeddings = self._encode_patterns()
            print(f"Loaded {len(self.patterns)} patterns with {self.encoder_dim}-dim embeddings")
        else:
            self.encoder = None
            self.pattern_embeddings = None
            print(f"Loaded {len(self.patterns)} patterns (no embeddings - install sentence-transformers)")

    def _load_library(self, path: str) -> dict:
        """加载JSON格式的模式库"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _encode_patterns(self) -> torch.Tensor:
        """预计算所有模式的嵌入向量"""
        texts = []
        for p in self.patterns:
            # 使用 name + description 作为编码输入
            text = f"{p['name']}: {p['description']}"
            texts.append(text)

        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.to(self.device)

    def match_patterns(self,
                       query_text: str,
                       top_k: int = 5,
                       category: str = None) -> List[Dict]:
        """
        匹配与查询最相关的对抗模式

        Args:
            query_text: 用户查询文本
            top_k: 返回最相关的k个模式
            category: 可选，限定在某一类别内搜索

        Returns:
            匹配结果列表，每项包含 pattern 和 score
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not available. Install sentence-transformers.")

        # 编码查询
        query_emb = self.encoder.encode(
            [query_text],
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(self.device)

        # 计算相似度
        if category:
            # 过滤特定类别
            indices = [i for i, p in enumerate(self.patterns) if p['category'] == category]
            if not indices:
                return []
            subset_embeddings = self.pattern_embeddings[indices]
            similarities = torch.mm(query_emb, subset_embeddings.t())[0]
            top_scores, top_local_indices = torch.topk(similarities, k=min(top_k, len(indices)))
            top_indices = [indices[i] for i in top_local_indices.tolist()]
        else:
            similarities = torch.mm(query_emb, self.pattern_embeddings.t())[0]
            top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(self.patterns)))
            top_indices = top_indices.tolist()
            top_scores = top_scores.tolist()

        # 构建结果
        results = []
        for idx, score in zip(top_indices, top_scores if isinstance(top_scores, list) else top_scores.tolist()):
            results.append({
                'pattern': self.patterns[idx],
                'score': score
            })

        return results

    def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict]:
        """根据ID获取特定模式"""
        for p in self.patterns:
            if p['id'] == pattern_id:
                return p
        return None

    def get_patterns_by_category(self, category: str) -> List[Dict]:
        """获取某一类别的所有模式"""
        return [p for p in self.patterns if p['category'] == category]

    def get_pattern_embedding(self, pattern_id: str) -> Optional[torch.Tensor]:
        """获取特定模式的嵌入向量"""
        for i, p in enumerate(self.patterns):
            if p['id'] == pattern_id:
                return self.pattern_embeddings[i]
        return None

    def get_all_pattern_texts(self) -> List[str]:
        """获取所有模式的文本描述（用于其他编码器）"""
        return [f"{p['name']}: {p['description']}" for p in self.patterns]

    def get_category_info(self) -> Dict:
        """获取所有类别信息"""
        return self.categories

    def summary(self) -> str:
        """返回模式库摘要信息"""
        category_counts = {}
        for p in self.patterns:
            cat = p['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1

        lines = [
            f"=== 对抗模式库摘要 ===",
            f"版本: {self.metadata['version']}",
            f"总模式数: {self.metadata['total_patterns']}",
            f"类别数: {self.metadata['categories']}",
            f"",
            f"各类别模式数量:"
        ]
        for cat_id, cat_info in self.categories.items():
            count = category_counts.get(cat_id, 0)
            lines.append(f"  - {cat_info['name']}: {count}个")

        return "\n".join(lines)


class AdversarialIntentEncoder(nn.Module):
    """
    对抗知识增强的意图编码器

    将用户查询与对抗模式库融合，生成增强型意图向量
    """

    def __init__(self,
                 intent_dim: int = 256,
                 library_path: str = None,
                 encoder_name: str = 'BAAI/bge-large-zh-v1.5',
                 num_attention_heads: int = 4,
                 dropout: float = 0.1):
        """
        初始化意图编码器

        Args:
            intent_dim: 输出的意图向量维度
            library_path: 对抗模式库路径
            encoder_name: 文本编码器名称
            num_attention_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()

        self.intent_dim = intent_dim

        # 加载对抗模式库
        self.pattern_library = AdversarialPatternLibrary(
            library_path=library_path,
            encoder_name=encoder_name
        )

        # 获取编码器维度
        if self.pattern_library.encoder is not None:
            self.encoder_dim = self.pattern_library.encoder_dim
        else:
            self.encoder_dim = 1024  # 默认BGE-large维度

        # 查询投影层
        self.query_projector = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder_dim // 2, intent_dim)
        )

        # 模式投影层
        self.pattern_projector = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder_dim // 2, intent_dim)
        )

        # 查询-模式注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=intent_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # 意图融合层
        self.intent_fusion = nn.Sequential(
            nn.Linear(intent_dim * 2, intent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intent_dim, intent_dim),
            nn.LayerNorm(intent_dim)
        )

        # 置信度估计器
        self.confidence_estimator = nn.Sequential(
            nn.Linear(intent_dim * 2, intent_dim // 2),
            nn.ReLU(),
            nn.Linear(intent_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self,
                query_text: str,
                top_k_patterns: int = 10,
                return_details: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        生成增强型意图向量

        Args:
            query_text: 用户查询文本
            top_k_patterns: 匹配的模式数量
            return_details: 是否返回详细信息

        Returns:
            intent_vector: 意图向量 [1, intent_dim]
            details: (可选) 包含匹配模式、置信度等信息
        """
        # Step 1: 编码查询
        query_emb = self.pattern_library.encoder.encode(
            [query_text],
            convert_to_tensor=True,
            normalize_embeddings=True
        )  # [1, encoder_dim]

        # Step 2: 匹配相关模式
        matched_results = self.pattern_library.match_patterns(query_text, top_k=top_k_patterns)
        matched_indices = [self.pattern_library.patterns.index(r['pattern']) for r in matched_results]
        matched_scores = torch.tensor([r['score'] for r in matched_results])

        # 获取匹配模式的嵌入
        pattern_embs = self.pattern_library.pattern_embeddings[matched_indices]  # [k, encoder_dim]

        # Step 3: 投影到意图空间
        query_proj = self.query_projector(query_emb)  # [1, intent_dim]
        pattern_proj = self.pattern_projector(pattern_embs)  # [k, intent_dim]

        # Step 4: 查询-模式注意力融合
        # 添加batch维度
        query_proj_3d = query_proj.unsqueeze(0)  # [1, 1, intent_dim]
        pattern_proj_3d = pattern_proj.unsqueeze(0)  # [1, k, intent_dim]

        attended_patterns, attention_weights = self.cross_attention(
            query_proj_3d,  # Query
            pattern_proj_3d,  # Key
            pattern_proj_3d   # Value
        )
        attended_patterns = attended_patterns.squeeze(0)  # [1, intent_dim]

        # Step 5: 融合生成意图向量
        concat_features = torch.cat([query_proj, attended_patterns], dim=-1)  # [1, intent_dim*2]
        intent_vector = self.intent_fusion(concat_features)  # [1, intent_dim]

        # Step 6: 估计置信度
        confidence = self.confidence_estimator(concat_features)  # [1, 1]

        if return_details:
            details = {
                'confidence': confidence.item(),
                'matched_patterns': [r['pattern'] for r in matched_results],
                'match_scores': matched_scores.tolist(),
                'attention_weights': attention_weights.squeeze(0).detach(),
                'query_embedding': query_emb.detach(),
                'intent_vector': intent_vector.detach()
            }
            return intent_vector, details

        return intent_vector, None


class SimpleIntentEncoder(nn.Module):
    """
    简化版意图编码器（不需要训练，直接使用）

    适用于快速原型验证
    """

    def __init__(self,
                 intent_dim: int = 256,
                 library_path: str = None,
                 encoder_name: str = 'BAAI/bge-large-zh-v1.5'):
        """
        初始化简化版意图编码器
        """
        super().__init__()

        self.intent_dim = intent_dim

        # 加载对抗模式库
        self.pattern_library = AdversarialPatternLibrary(
            library_path=library_path,
            encoder_name=encoder_name
        )

        self.encoder_dim = self.pattern_library.encoder_dim

        # 简单的线性投影
        self.projector = nn.Linear(self.encoder_dim, intent_dim)

    def forward(self,
                query_text: str,
                top_k_patterns: int = 5,
                temperature: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        生成意图向量（简化版）

        使用加权平均融合查询和匹配模式
        """
        # 编码查询
        query_emb = self.pattern_library.encoder.encode(
            [query_text],
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # 匹配模式
        matched_results = self.pattern_library.match_patterns(query_text, top_k=top_k_patterns)
        matched_indices = [self.pattern_library.patterns.index(r['pattern']) for r in matched_results]
        pattern_embs = self.pattern_library.pattern_embeddings[matched_indices]

        # 计算注意力权重
        similarities = torch.tensor([r['score'] for r in matched_results])
        attention_weights = F.softmax(similarities / temperature, dim=0)

        # 加权聚合模式嵌入
        weighted_patterns = torch.sum(
            attention_weights.unsqueeze(-1) * pattern_embs,
            dim=0,
            keepdim=True
        )

        # 融合查询和模式
        fused_emb = 0.5 * query_emb + 0.5 * weighted_patterns

        # 投影到意图空间
        intent_vector = self.projector(fused_emb)
        intent_vector = F.normalize(intent_vector, dim=-1)

        details = {
            'matched_patterns': [r['pattern']['name'] for r in matched_results],
            'match_scores': similarities.tolist(),
            'attention_weights': attention_weights.tolist()
        }

        return intent_vector, details


# ==================== 测试代码 ====================

def test_pattern_library():
    """测试对抗模式库"""
    print("\n" + "="*60)
    print("测试对抗模式库")
    print("="*60)

    # 加载模式库
    library = AdversarialPatternLibrary()

    # 打印摘要
    print(library.summary())

    # 测试查询匹配
    test_queries = [
        "找出在社交媒体上协同传播虚假信息的隐蔽群体",
        "检测批量注册的机器人账号",
        "识别使用暗号进行秘密通信的用户"
    ]

    print("\n--- 查询匹配测试 ---")
    for query in test_queries:
        print(f"\n查询: {query}")
        results = library.match_patterns(query, top_k=3)
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['pattern']['id']}] {r['pattern']['name']} (相似度: {r['score']:.3f})")


def test_intent_encoder():
    """测试意图编码器"""
    print("\n" + "="*60)
    print("测试意图编码器")
    print("="*60)

    # 初始化编码器
    encoder = SimpleIntentEncoder(intent_dim=256)

    # 测试编码
    query = "找出在社交网络上通过隐蔽连接协同传播极端内容的群体"
    intent_vector, details = encoder(query, top_k_patterns=5)

    print(f"\n查询: {query}")
    print(f"意图向量维度: {intent_vector.shape}")
    print(f"意图向量范数: {torch.norm(intent_vector).item():.4f}")
    print(f"\n匹配的对抗模式:")
    for name, score, weight in zip(details['matched_patterns'],
                                    details['match_scores'],
                                    details['attention_weights']):
        print(f"  - {name} (相似度: {score:.3f}, 权重: {weight:.3f})")


if __name__ == '__main__':
    # 运行测试
    test_pattern_library()

    if HAS_SENTENCE_TRANSFORMERS:
        test_intent_encoder()
    else:
        print("\n跳过意图编码器测试（需要安装sentence-transformers）")
