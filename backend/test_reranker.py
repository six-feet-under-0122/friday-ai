# test_reranker.py
import rag_core

# 测试查询
query = "大学生的情绪有什么特点？"

# 调用带重排的检索
context, sources = rag_core.retrieve_with_sources(query, k=3)

print("="*60)
print(f"查询: {query}")
print("="*60)
print("\n检索结果:\n")
for i, source in enumerate(sources):
    print(f"[{i+1}] 来源: {source}")
print(f"\n上下文长度: {len(context)} 字符")